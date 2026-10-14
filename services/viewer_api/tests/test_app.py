from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import services.viewer_api.app as viewer_app
from services.viewer_api.app import create_app
from services.viewer_api.config import ViewerSettings

PUBLICATION_DIGEST = "a" * 64
PUBLICATION_ID = f"publication:sha256:{PUBLICATION_DIGEST}"


def write_tileset(root: Path, *, content_count: int = 20) -> None:
    root.mkdir(parents=True)
    children = []
    for index in range(content_count):
        children.append(
            {
                "boundingVolume": {"sphere": [index, 0, 0, 1]},
                "content": {"uri": f"tiles/{index}.glb"},
                "geometricError": 0,
            }
        )
    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": 100,
        "root": {
            "boundingVolume": {"sphere": [0, 0, 0, 100]},
            "children": children,
            "geometricError": 50,
            "refine": "REPLACE",
        },
    }
    (root / "tileset.json").write_text(json.dumps(tileset), encoding="utf-8")


def create_settings(tmp_path: Path) -> ViewerSettings:
    viewer_dist = tmp_path / "dist"
    viewer_dist.mkdir()
    (viewer_dist / "index.html").write_text("<h1>viewer</h1>", encoding="utf-8")

    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    write_tileset(left_root, content_count=1)
    write_tileset(right_root)
    tile_root = right_root / "tiles"
    tile_root.mkdir()
    (tile_root / "0.glb").write_bytes(b"0123456789")
    return ViewerSettings(
        viewer_dist=viewer_dist,
        left_root=left_root,
        right_root=right_root,
        left_label="Left",
        right_label="Right",
        port=9000,
    )


def create_publication_settings(tmp_path: Path) -> ViewerSettings:
    settings = create_settings(tmp_path)
    publications = tmp_path / "publications"
    publication = publications / PUBLICATION_DIGEST
    write_tileset(publication, content_count=1)
    tile_root = publication / "tiles"
    tile_root.mkdir()
    (tile_root / "0.glb").write_bytes(b"0123456789")
    (publication / "source-leaf-manifest.json").write_text(
        json.dumps({"inputObj": "/private/worker/job/model.obj"}),
        encoding="utf-8",
    )
    registry = tmp_path / "publications.sqlite3"
    manifest = {
        "publicationId": PUBLICATION_ID,
        "atomicDirectory": {"directoryName": PUBLICATION_DIGEST},
        "tileset": {"relativePath": "tileset.json"},
    }
    with closing(sqlite3.connect(registry)) as connection:
        connection.execute(
            "CREATE TABLE publications (publication_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO publications(publication_id, manifest_json) VALUES(?, ?)",
            (PUBLICATION_ID, json.dumps(manifest)),
        )
        connection.commit()
    return replace(
        settings,
        publications_root=publications,
        publication_registry=registry,
        publication_bearer_token="test-publication-token",
        publication_cors_origins=("https://viewer.example",),
    )


def count_contents(tile: dict[str, object]) -> int:
    children = tile.get("children", [])
    return int("content" in tile) + sum(count_contents(child) for child in children)


@pytest.mark.anyio
async def test_config_previews_and_static_range_requests(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(create_settings(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        config = await client.get("/api/config")
        assert config.status_code == 200
        assert config.json()["maximumScreenSpaceError"] == 8
        assert config.json()["models"]["a1"]["label"] == "Right"

        preview = (await client.get("/api/a1-preview")).json()
        assert preview["root"]["content"]["uri"] == "/a1/tiles/0.glb"
        assert preview["root"]["geometricError"] == 100
        medium = (await client.get("/api/a1-medium")).json()
        assert count_contents(medium["root"]) == 17

        response = await client.get("/a1/tiles/0.glb", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"
        assert response.headers["content-type"] == "model/gltf-binary"
        assert (await client.get("/")).text == "<h1>viewer</h1>"


@pytest.mark.anyio
async def test_diagnostics_are_validated_and_round_trip(tmp_path: Path) -> None:
    payload = {
        "activeView": "overview",
        "mode": "a1",
        "pending": 1,
        "processing": 2,
        "camera": {"position": [1, 2, 3], "fov": 1.0, "viewportHeight": 760},
        "visible": [
            {
                "contentUrl": "/a1/tiles/0.glb",
                "triangles": 42,
                "geometricError": 0,
                "depth": 1,
                "center": [1, 2, 3],
                "radius": 4,
            }
        ],
        "runtime": {
            "version": 56,
            "metricDomWrites": 12,
            "resourceBatches": 3,
            "fullResourceScans": 0,
            "hiddenSuspensions": 1,
            "resourceMode": "observer",
            "cumulativeResourceEntries": 20,
            "ignoredResourceEntries": 0,
            "cadenceMs": 250,
            "idle": False,
            "hidden": False,
            "resourceCountersRetainTimingEntries": False,
            "limitations": ["engine estimate"],
        },
        "cache": {
            "policy": "convergent-demand-v55",
            "loads": 10,
            "unloads": 2,
            "reloads": 1,
            "trims": 1,
            "trimDecision": "under-soft-budget",
            "epoch": 2,
            "convergencePasses": 1,
            "convergenceStopReason": None,
            "settings": {
                "modelKey": "a1",
                "resourcePrefix": "/a1/",
                "retainMs": 30000,
                "softBytes": 268435456,
                "hardBytes": 1073741824,
            },
            "enabled": True,
            "pending": 1,
            "processing": 2,
            "failures": 0,
            "visibleTiles": 8,
            "loadedTiles": 12,
            "nonVisibleCachedTiles": 4,
            "memoryBytes": 123456,
            "resource": {
                "requests": 20,
                "encodedBodyBytes": 1000,
                "decodedBodyBytes": 2000,
                "transferBytes": 900,
            },
            "activeView": "overview",
            "mode": "a1",
            "capturedAt": "2026-09-01T00:00:00Z",
        },
        "capturedAt": "2026-09-01T00:00:00Z",
    }
    transport = httpx.ASGITransport(app=create_app(create_settings(tmp_path)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/diagnostics")).json() == {"status": "waiting-for-browser"}
        assert (await client.post("/api/diagnostics", json=payload)).json() == {"status": "ok"}
        assert (await client.get("/api/diagnostics")).json() == payload


@pytest.mark.anyio
async def test_authorized_publication_route_supports_relative_range_and_cors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_publication_settings(tmp_path)
    registry_opens = 0
    connect = viewer_app.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        nonlocal registry_opens
        registry_opens += 1
        return connect(*args, **kwargs)

    monkeypatch.setattr(viewer_app.sqlite3, "connect", tracked_connect)
    transport = httpx.ASGITransport(app=create_app(settings))
    base = f"/v1/mesh-publications/{PUBLICATION_ID}"
    authorization = {"Authorization": "Bearer test-publication-token"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.get(f"{base}/tileset.json")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        wrong_token = await client.get(
            f"{base}/tileset.json",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert wrong_token.status_code == 401

        tileset = await client.get(f"{base}/tileset.json", headers=authorization)
        assert tileset.status_code == 200
        assert tileset.json()["root"]["children"][0]["content"]["uri"] == "tiles/0.glb"
        assert str(tmp_path) not in tileset.text

        content = await client.get(
            f"{base}/tiles/0.glb",
            headers={
                **authorization,
                "Range": "bytes=2-5",
                "Origin": "https://viewer.example",
            },
        )
        assert content.status_code == 206
        assert content.content == b"2345"
        assert content.headers["content-range"] == "bytes 2-5/10"
        assert content.headers["access-control-allow-origin"] == "https://viewer.example"
        head = await client.head(f"{base}/tiles/0.glb", headers=authorization)
        assert head.status_code == 200
        assert head.headers["content-length"] == "10"
        assert head.content == b""
        assert registry_opens == 1

        preflight = await client.options(
            f"{base}/tileset.json",
            headers={
                "Origin": "https://viewer.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,range",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://viewer.example"
        assert "Authorization" in preflight.headers["access-control-allow-headers"]

        missing = await client.get(
            f"/v1/mesh-publications/publication:sha256:{'b' * 64}/tileset.json",
            headers=authorization,
        )
        assert missing.status_code == 404
        assert str(tmp_path) not in missing.text
        temporary = await client.get(
            "/v1/mesh-publications/.tmp-incomplete/tileset.json",
            headers=authorization,
        )
        assert temporary.status_code == 404
        unreferenced = await client.get(
            f"{base}/source-leaf-manifest.json",
            headers=authorization,
        )
        assert unreferenced.status_code == 404
        assert "/private/worker" not in unreferenced.text
        traversal = await client.get(
            f"{base}/%2e%2e/tileset.json",
            headers=authorization,
        )
        assert traversal.status_code == 404


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
