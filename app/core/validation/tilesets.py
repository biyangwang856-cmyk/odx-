#!/usr/bin/env python3
"""Audit candidate publications and select the production Mesh default."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

MAX_JSON_BYTES = 16 * 1024 * 1024


@dataclass
class Scan:
    root: Path
    tilesets: set[Path] = field(default_factory=set)
    contents: list[Path] = field(default_factory=list)
    extensions: Counter[str] = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)


def _finite_nonnegative(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _resolve_reference(scan: Scan, parent: Path, uri: object) -> Path | None:
    if not isinstance(uri, str) or not uri:
        scan.failures.append("content URI is missing")
        return None
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or parsed.path.startswith("/"):
        scan.failures.append(f"content URI is not a relative local path: {uri}")
        return None
    try:
        candidate = (parent / unquote(parsed.path)).resolve()
    except (OSError, RuntimeError):
        scan.failures.append(f"content URI cannot be resolved: {uri}")
        return None
    try:
        candidate.relative_to(scan.root)
    except ValueError:
        scan.failures.append(f"content URI escapes publication root: {uri}")
        return None
    if not candidate.is_file():
        scan.failures.append(f"content reference is missing: {candidate.relative_to(scan.root)}")
        return None
    return candidate


def _content_uri(content: object) -> object:
    if not isinstance(content, dict):
        return None
    return content.get("uri", content.get("url"))


def _tileset_document_failures(document: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(document, dict):
        return ["tileset document is not an object"]
    asset = document.get("asset")
    if (
        not isinstance(asset, dict)
        or not isinstance(asset.get("version"), str)
        or not asset["version"].strip()
    ):
        failures.append("tileset asset version is invalid")
    if not _finite_nonnegative(document.get("geometricError")):
        failures.append("tileset geometricError is invalid")
    root_tile = document.get("root")
    if not isinstance(root_tile, dict):
        failures.append("tileset root is missing")
        return failures

    def visit_tile(tile: object, location: str) -> None:
        if not isinstance(tile, dict):
            failures.append(f"{location} is not an object")
            return
        if not isinstance(tile.get("boundingVolume"), dict):
            failures.append(f"{location} boundingVolume is missing")
        if not _finite_nonnegative(tile.get("geometricError")):
            failures.append(f"{location} geometricError is invalid")
        if "content" in tile:
            uri = _content_uri(tile["content"])
            if not isinstance(uri, str) or not uri:
                failures.append(f"{location} content URI is missing")
        contents = tile.get("contents", [])
        if not isinstance(contents, list):
            failures.append(f"{location} contents is not an array")
        else:
            for index, content in enumerate(contents):
                uri = _content_uri(content)
                if not isinstance(uri, str) or not uri:
                    failures.append(f"{location}.contents[{index}] URI is missing")
        children = tile.get("children", [])
        if not isinstance(children, list):
            failures.append(f"{location} children is not an array")
            return
        for index, child in enumerate(children):
            visit_tile(child, f"{location}.children[{index}]")

    visit_tile(root_tile, "root")
    return failures


def _visit_tileset(scan: Scan, path: Path) -> None:
    if path in scan.tilesets:
        return
    scan.tilesets.add(path)
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("tileset JSON exceeds size limit")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        scan.failures.append(f"invalid tileset {path.relative_to(scan.root)}: {error}")
        return
    relative_tileset = path.relative_to(scan.root)
    scan.failures.extend(f"{relative_tileset}: {failure}" for failure in _tileset_document_failures(document))
    if not isinstance(document, dict):
        return
    root_tile = document.get("root")
    if not isinstance(root_tile, dict):
        return

    def visit_tile(tile: object) -> None:
        if not isinstance(tile, dict):
            return
        content_values: list[object] = []
        if "content" in tile:
            content_values.append(tile["content"])
        contents = tile.get("contents", [])
        if isinstance(contents, list):
            content_values.extend(contents)
        elif "contents" in tile:
            contents = []
        for content in content_values:
            uri = _content_uri(content)
            if not isinstance(uri, str) or not uri:
                continue
            resolved = _resolve_reference(scan, path.parent, uri)
            if resolved is None:
                continue
            if resolved.suffix.lower() == ".json":
                _visit_tileset(scan, resolved)
                continue
            scan.contents.append(resolved)
            suffix = resolved.suffix.lower()
            scan.extensions[suffix] += 1
            expected_magic = {".glb": b"glTF", ".b3dm": b"b3dm"}.get(suffix)
            if expected_magic:
                try:
                    with resolved.open("rb") as stream:
                        magic = stream.read(4)
                except OSError as error:
                    scan.failures.append(
                        f"content cannot be read: {resolved.relative_to(scan.root)}: {error}"
                    )
                    continue
                if magic != expected_magic:
                    scan.failures.append(f"invalid {suffix} magic: {resolved.relative_to(scan.root)}")
        children = tile.get("children", [])
        if not isinstance(children, list):
            return
        for child in children:
            visit_tile(child)

    visit_tile(root_tile)


def _scan_filesystem(root: Path) -> tuple[dict[str, object], Scan]:
    try:
        resolved = root.resolve()
    except (OSError, RuntimeError):
        resolved = root.absolute()
    scan = Scan(root=resolved)
    if root.is_symlink() or not resolved.is_dir():
        scan.failures.append("publication root is missing or symbolic")
    else:
        entrypoint = resolved / "tileset.json"
        if not entrypoint.is_file():
            scan.failures.append("root tileset.json is missing")
        else:
            _visit_tileset(scan, entrypoint)
    report = {
        "passed": not scan.failures,
        "tilesets": len(scan.tilesets),
        "contentReferences": len(scan.contents),
        "uniqueContents": len(set(scan.contents)),
        "extensions": dict(sorted(scan.extensions.items())),
        "failures": scan.failures,
    }
    return report, scan


def audit_filesystem(root: Path) -> dict[str, object]:
    """Validate every tileset and local content reference below one entrypoint."""
    return scan_filesystem(root)[0]


def scan_filesystem(root: Path) -> tuple[dict[str, object], Scan]:
    """Return the filesystem report and its verified referenced paths."""
    return _scan_filesystem(root)


def _cors_passes(headers: Any, origin: str) -> bool:
    return headers.get("Access-Control-Allow-Origin") in {"*", origin}


def _audit_http(
    scan: Scan,
    base_url: str,
    origin: str,
    *,
    opener: Any,
) -> dict[str, object]:
    failures: list[str] = []
    json_requests = 0
    range_requests = 0
    for path in sorted(scan.tilesets):
        relative = path.relative_to(scan.root).as_posix()
        url = urljoin(base_url.rstrip("/") + "/", relative)
        request = Request(url, headers={"Accept": "application/json", "Origin": origin})
        try:
            with opener(request, timeout=10) as response:
                body = response.read(MAX_JSON_BYTES + 1)
                if response.status != 200 or len(body) > MAX_JSON_BYTES:
                    failures.append(f"JSON request failed: {relative}")
                elif not _cors_passes(response.headers, origin):
                    failures.append(f"CORS header failed: {relative}")
                else:
                    document_failures = _tileset_document_failures(json.loads(body))
                    failures.extend(
                        f"HTTP tileset is invalid: {relative}: {failure}" for failure in document_failures
                    )
        except (HTTPError, URLError, IncompleteRead, OSError, TimeoutError, ValueError) as error:
            failures.append(f"JSON request failed: {relative}: {error}")
        json_requests += 1

    for path in sorted(set(scan.contents)):
        relative = path.relative_to(scan.root).as_posix()
        url = urljoin(base_url.rstrip("/") + "/", relative)
        request = Request(
            url,
            headers={"Origin": origin, "Range": "bytes=0-3"},
        )
        try:
            with opener(request, timeout=10) as response:
                body = response.read(5)
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or len(body) != 4 or not content_range.startswith("bytes 0-3/"):
                    failures.append(f"Range request failed: {relative}")
                elif not _cors_passes(response.headers, origin):
                    failures.append(f"CORS header failed: {relative}")
                else:
                    expected_magic = {".glb": b"glTF", ".b3dm": b"b3dm"}.get(path.suffix.lower())
                    if expected_magic and body != expected_magic:
                        failures.append(f"HTTP content magic failed: {relative}")
        except (HTTPError, URLError, IncompleteRead, OSError, TimeoutError, ValueError) as error:
            failures.append(f"Range request failed: {relative}: {error}")
        range_requests += 1
    return {
        "passed": not failures,
        "jsonRequests": json_requests,
        "rangeRequests": range_requests,
        "failures": failures,
    }


def audit_candidate(
    candidate_id: str,
    root: Path,
    base_url: str,
    origin: str,
    *,
    opener: Any = urlopen,
) -> dict[str, object]:
    filesystem, scan = _scan_filesystem(root)
    http = _audit_http(scan, base_url, origin, opener=opener)
    return {
        "schemaVersion": "mesh-candidate-acceptance.v1",
        "candidateId": candidate_id,
        "passed": bool(filesystem["passed"] and http["passed"]),
        "filesystem": filesystem,
        "http": http,
    }
