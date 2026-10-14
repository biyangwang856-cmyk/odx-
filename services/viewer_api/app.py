"""FastAPI application and process entrypoint for the Cesium result viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import secrets
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import TARGETS, ViewerSettings
from .models import DiagnosticsPayload
from .tilesets import JsonObject, create_medium_tileset, create_preview_tileset, read_tileset

PUBLICATION_ID = re.compile(r"publication:sha256:([0-9a-f]{64})\Z")

for media_type, extension in (
    ("application/octet-stream", ".b3dm"),
    ("model/gltf-binary", ".glb"),
    ("image/ktx2", ".ktx2"),
    ("application/wasm", ".wasm"),
    ("image/webp", ".webp"),
):
    mimetypes.add_type(media_type, extension)


def create_app(settings: ViewerSettings | None = None) -> FastAPI:
    """Create one configured service without importing the research task package."""
    resolved = settings or ViewerSettings.for_target()
    resolved.validate()
    source_tileset = read_tileset(resolved.right_root)
    preview_tileset = create_preview_tileset(source_tileset)
    medium_tileset = create_medium_tileset(source_tileset)

    app = FastAPI(title="Mesh Research Viewer API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.publication_cors_origins),
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Authorization", "Range"],
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    )
    app.state.latest_diagnostics = None

    def require_publication_authorization(authorization: str | None) -> None:
        token = resolved.publication_bearer_token
        if token is None:
            raise HTTPException(status_code=503, detail="publication service is not configured")
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(
                status_code=401,
                detail="invalid publication authorization",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def publication_directory(publication_id: str) -> tuple[Path, Mapping[str, Any]]:
        match = PUBLICATION_ID.fullmatch(publication_id)
        if match is None:
            raise HTTPException(status_code=404, detail="publication not found")
        if resolved.publications_root is None or resolved.publication_registry is None:
            raise HTTPException(status_code=503, detail="publication service is not configured")
        try:
            with closing(
                sqlite3.connect(f"{resolved.publication_registry.as_uri()}?mode=ro", uri=True)
            ) as connection:
                row = connection.execute(
                    "SELECT manifest_json FROM publications WHERE publication_id=?",
                    (publication_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise HTTPException(status_code=503, detail="publication registry unavailable") from error
        if row is None:
            raise HTTPException(status_code=404, detail="publication not found")
        try:
            manifest = json.loads(row[0])
            directory_name = manifest["atomicDirectory"]["directoryName"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise HTTPException(status_code=503, detail="publication registry entry is invalid") from error
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("publicationId") != publication_id
            or directory_name != match.group(1)
        ):
            raise HTTPException(status_code=503, detail="publication registry entry is invalid")
        directory = resolved.publications_root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise HTTPException(status_code=404, detail="publication content not found")
        return directory, manifest

    def publication_assets(directory: Path, manifest: Mapping[str, Any]) -> set[str]:
        try:
            root_tileset = manifest["tileset"]["relativePath"]
        except (KeyError, TypeError) as error:
            raise HTTPException(status_code=503, detail="publication registry entry is invalid") from error
        if not isinstance(root_tileset, str):
            raise HTTPException(status_code=503, detail="publication registry entry is invalid")
        root_relative = PurePosixPath(root_tileset)
        if (
            root_relative.is_absolute()
            or "\\" in root_tileset
            or any(part in {"", ".", ".."} for part in root_relative.parts)
        ):
            raise HTTPException(status_code=503, detail="publication registry entry is invalid")
        pending = [root_relative]
        allowed: set[str] = set()
        visited_tilesets: set[str] = set()

        while pending:
            tileset_relative = pending.pop()
            tileset_name = tileset_relative.as_posix()
            if tileset_name in visited_tilesets:
                continue
            visited_tilesets.add(tileset_name)
            allowed.add(tileset_name)
            try:
                document = json.loads((directory / tileset_relative).read_text(encoding="utf-8"))
                root = document["root"]
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise HTTPException(status_code=503, detail="published tileset is invalid") from error

            tiles = [root]
            while tiles:
                tile = tiles.pop()
                if not isinstance(tile, Mapping):
                    raise HTTPException(status_code=503, detail="published tileset is invalid")
                contents = []
                if "content" in tile:
                    contents.append(tile["content"])
                multiple = tile.get("contents", [])
                if not isinstance(multiple, list):
                    raise HTTPException(status_code=503, detail="published tileset is invalid")
                contents.extend(multiple)
                for content in contents:
                    if not isinstance(content, Mapping):
                        raise HTTPException(status_code=503, detail="published tileset is invalid")
                    uri = content.get("uri", content.get("url"))
                    if not isinstance(uri, str):
                        raise HTTPException(status_code=503, detail="published tileset is invalid")
                    relative = tileset_relative.parent / PurePosixPath(uri)
                    if (
                        relative.is_absolute()
                        or "\\" in uri
                        or any(part in {"", ".", ".."} for part in relative.parts)
                    ):
                        raise HTTPException(status_code=503, detail="published tileset is invalid")
                    allowed.add(relative.as_posix())
                    if relative.suffix.lower() == ".json":
                        pending.append(relative)
                children = tile.get("children", [])
                if not isinstance(children, list):
                    raise HTTPException(status_code=503, detail="published tileset is invalid")
                tiles.extend(children)
        return allowed

    @lru_cache(maxsize=64)
    def published_assets(publication_id: str) -> tuple[Path, frozenset[str]]:
        directory, manifest = publication_directory(publication_id)
        return directory, frozenset(publication_assets(directory, manifest))

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def viewer_config() -> dict[str, Any]:
        return {
            "dataset": resolved.dataset,
            "maximumScreenSpaceError": resolved.maximum_screen_space_error,
            "models": {
                "terra": {"label": resolved.left_label, "tilesetUrl": "/terra/tileset.json"},
                "a1": {"label": resolved.right_label, "tilesetUrl": "/a1/tileset.json"},
            },
        }

    @app.get("/api/diagnostics")
    def read_diagnostics() -> dict[str, Any]:
        return app.state.latest_diagnostics or {"status": "waiting-for-browser"}

    @app.post("/api/diagnostics")
    def write_diagnostics(payload: DiagnosticsPayload) -> dict[str, str]:
        app.state.latest_diagnostics = payload.model_dump(by_alias=True, mode="json")
        return {"status": "ok"}

    @app.get("/api/a1-preview", response_class=JSONResponse)
    def a1_preview() -> JsonObject:
        return preview_tileset

    @app.get("/api/a1-medium", response_class=JSONResponse)
    def a1_medium() -> JsonObject:
        return medium_tileset

    @app.api_route(
        "/v1/mesh-publications/{publication_id}/{asset_path:path}",
        methods=["GET", "HEAD"],
        response_class=FileResponse,
    )
    def publication_asset(
        publication_id: str,
        asset_path: str,
        authorization: str | None = Header(default=None),
    ) -> FileResponse:
        require_publication_authorization(authorization)
        directory, allowed = published_assets(publication_id)
        relative = PurePosixPath(asset_path)
        if (
            not asset_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise HTTPException(status_code=404, detail="publication content not found")
        if relative.as_posix() not in allowed:
            raise HTTPException(status_code=404, detail="publication content not found")
        target = directory.joinpath(*relative.parts)
        cursor = directory
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise HTTPException(status_code=404, detail="publication content not found")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="publication content not found")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(
            target,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    app.mount("/terra", StaticFiles(directory=resolved.left_root), name="terra")
    app.mount("/a1", StaticFiles(directory=resolved.right_root), name="a1")
    app.mount("/", StaticFiles(directory=resolved.viewer_dist, html=True), name="viewer")
    return app


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one built Cesium comparison publication.")
    parser.add_argument("--target", choices=sorted(TARGETS), default="v55")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings = ViewerSettings.for_target(arguments.target)
    settings.validate()
    uvicorn.run(
        create_app(settings),
        host=arguments.host,
        port=arguments.port or settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
