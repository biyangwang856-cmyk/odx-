#!/usr/bin/env python3
"""Map visible HLOD proxies back to their original V17 subtrees.

The script intentionally reads only tileset JSON and the JSON chunk at the
front of GLB files.  It does not decode textures or mesh buffers, so it is
safe to run against a large publication on JuiceFS.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROXY_RE = re.compile(r"mesh-(\d{2})-(\d{3})\.glb$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-count", type=int, default=12)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--leaf-samples", type=int, default=3)
    return parser.parse_args()


def content_uri(tile: dict[str, Any]) -> str | None:
    content = tile.get("content") or {}
    return content.get("uri") or content.get("url")


def walk_parallel(
    candidate: dict[str, Any],
    original: dict[str, Any],
    path: tuple[int, ...] = (),
) -> Iterable[tuple[tuple[int, ...], dict[str, Any], dict[str, Any]]]:
    yield path, candidate, original
    c_children = candidate.get("children") or []
    o_children = original.get("children") or []
    if len(c_children) != len(o_children):
        raise ValueError(
            f"tree mismatch at {'.'.join(map(str, path)) or 'root'}: "
            f"candidate={len(c_children)}, original={len(o_children)}"
        )
    for index, (c_child, o_child) in enumerate(zip(c_children, o_children)):
        yield from walk_parallel(c_child, o_child, path + (index,))


def collect_leaf_uris(tile: dict[str, Any]) -> list[str]:
    children = tile.get("children") or []
    if not children:
        uri = content_uri(tile)
        return [uri] if uri else []
    result: list[str] = []
    for child in children:
        result.extend(collect_leaf_uris(child))
    return result


def resolve_content(tileset_path: Path, uri: str) -> Path | None:
    if not uri or "://" in uri or uri.startswith("data:"):
        return None
    return (tileset_path.parent / uri.split("?", 1)[0]).resolve()


def inspect_glb(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": bool(path and path.exists()),
    }
    if not path or not path.exists():
        return result
    result["bytes"] = path.stat().st_size
    try:
        with path.open("rb") as handle:
            header = handle.read(20)
            if len(header) < 20 or header[:4] != b"glTF":
                result["format"] = "not-glb"
                return result
            _, version, total_length, json_length, json_type = struct.unpack("<4sIIII", header)
            if json_type != 0x4E4F534A:
                raise ValueError(f"first GLB chunk is not JSON: {json_type:#x}")
            document = json.loads(handle.read(json_length).decode("utf-8"))
        attributes: Counter[str] = Counter()
        primitive_count = 0
        for mesh in document.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                primitive_count += 1
                attributes.update((primitive.get("attributes") or {}).keys())
        extensions = document.get("extensionsUsed") or []
        result.update(
            {
                "format": "glb",
                "version": version,
                "declaredBytes": total_length,
                "meshes": len(document.get("meshes", [])),
                "primitives": primitive_count,
                "images": len(document.get("images", [])),
                "textures": len(document.get("textures", [])),
                "materials": len(document.get("materials", [])),
                "attributes": dict(attributes),
                "extensionsUsed": extensions,
                "hasTexcoord": any(name.startswith("TEXCOORD_") for name in attributes),
                "hasVertexColor": "COLOR_0" in attributes,
                "hasTexture": bool(document.get("images") or document.get("textures")),
                "unlit": "KHR_materials_unlit" in extensions,
            }
        )
    except Exception as error:  # retain useful path/size evidence
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def load_proxy_metrics(candidate_root: Path, shard: str) -> dict[str, Any]:
    path = candidate_root / f"intermediate-mesh-hlod-shard-{shard}-report.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {node["proxy"]: node for node in report.get("nodes", [])}


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    camera_position = audit["camera"]["position"]
    fov = float(audit["camera"]["frustum"]["fov"])
    focal_pixels = args.viewport_height / (2.0 * math.tan(fov / 2.0))

    visible_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for visible in audit.get("visible", []):
        name = Path(visible.get("contentUrl") or "").name
        match = PROXY_RE.search(name)
        if match:
            visible_by_shard[match.group(1)].append(visible)

    rows: list[dict[str, Any]] = []
    for shard, visible_rows in sorted(visible_by_shard.items()):
        c_tileset = args.candidate_root / "detail" / "shards" / shard / "tileset.json"
        o_tileset = args.original_root / "detail" / "shards" / shard / "tileset.json"
        candidate = json.loads(c_tileset.read_text(encoding="utf-8"))
        original = json.loads(o_tileset.read_text(encoding="utf-8"))
        by_proxy: dict[str, tuple[tuple[int, ...], dict[str, Any], dict[str, Any]]] = {}
        for path, c_node, o_node in walk_parallel(candidate["root"], original["root"]):
            uri = content_uri(c_node)
            if uri and PROXY_RE.search(Path(uri).name):
                by_proxy[Path(uri).name] = (path, c_node, o_node)
        metrics = load_proxy_metrics(args.candidate_root, shard)

        for visible in visible_rows:
            name = Path(visible["contentUrl"]).name
            if name not in by_proxy:
                raise KeyError(f"visible proxy not found in shard {shard}: {name}")
            path, c_node, o_node = by_proxy[name]
            leaf_uris = collect_leaf_uris(o_node)
            center = visible["center"]
            distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(center, camera_position)))
            radius = float(visible["radius"])
            screen_radius = focal_pixels * radius / max(distance, 1e-9)
            proxy_path = args.candidate_root / "proxy" / name
            sample_paths = [resolve_content(o_tileset, uri) for uri in leaf_uris[: args.leaf_samples]]
            row_metrics = metrics.get(name) or {}
            measured = row_metrics.get("measuredError") or {}
            rows.append(
                {
                    "proxy": name,
                    "shard": shard,
                    "treePath": list(path),
                    "runtimePath": visible.get("runtimePath"),
                    "depth": visible.get("depth"),
                    "children": len(c_node.get("children") or []),
                    "descendantLeaves": len(leaf_uris),
                    "originalLeafUris": leaf_uris,
                    "candidateGeometricErrorM": c_node.get("geometricError"),
                    "originalGeometricErrorM": o_node.get("geometricError"),
                    "distanceM": distance,
                    "boundingRadiusM": radius,
                    "projectedRadiusPx": screen_radius,
                    "projectedAreaScore": math.pi * screen_radius * screen_radius,
                    "trianglesAtRuntime": visible.get("triangles"),
                    "measuredP99M": measured.get("areaWeightedP99M"),
                    "measuredMaximumM": measured.get("maximumM"),
                    "surfaceAreaRatio": measured.get("candidateToSourceAreaRatio"),
                    "proxyGlb": inspect_glb(proxy_path),
                    "originalLeafSamples": [inspect_glb(p) for p in sample_paths],
                }
            )

    rows.sort(key=lambda row: row["projectedAreaScore"], reverse=True)
    selected = rows[: max(0, min(args.top_count, len(rows)))]
    selected_names = {row["proxy"] for row in selected}
    for rank, row in enumerate(rows, start=1):
        row["screenFootprintRank"] = rank
        row["selectedForTargetedV7"] = row["proxy"] in selected_names

    proxy_texture_counts = Counter(
        "textured" if row["proxyGlb"].get("hasTexture") else "untextured" for row in rows
    )
    leaf_sample_counts = Counter()
    for row in rows:
        for sample in row["originalLeafSamples"]:
            leaf_sample_counts["textured" if sample.get("hasTexture") else "untextured"] += 1

    output = {
        "audit": str(args.audit),
        "candidateRoot": str(args.candidate_root),
        "originalRoot": str(args.original_root),
        "selectionPolicy": {
            "method": "largest projected bounding-sphere footprint in the fixed texture camera",
            "topCount": args.top_count,
            "selected": [row["proxy"] for row in selected],
            "selectedShards": sorted({row["shard"] for row in selected}),
        },
        "summary": {
            "visibleProxyCount": len(rows),
            "shards": dict(sorted(Counter(row["shard"] for row in rows).items())),
            "proxyTextureInspection": dict(proxy_texture_counts),
            "originalLeafSampleTextureInspection": dict(leaf_sample_counts),
            "allVisibleContentIsProxy": len(rows) == len(audit.get("visible", [])),
            "stablePending": audit.get("pending"),
            "stableProcessing": audit.get("processing"),
            "consoleErrors": len(audit.get("consoleErrors") or []),
            "failedRequests": len(audit.get("failedRequests") or []),
        },
        "visibleProxies": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print(json.dumps(output["selectionPolicy"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
