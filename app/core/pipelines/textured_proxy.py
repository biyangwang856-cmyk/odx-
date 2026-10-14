#!/usr/bin/env python3
"""Build a small UV-preserving textured-proxy POC from original V17 leaves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--selected-start-rank", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--error", type=float, default=0.001)
    parser.add_argument("--lock-border", choices=("true", "false"), default="true")
    return parser.parse_args()


def json_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def clone_publication(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    def copy_json_or_link(src: str, dst: str) -> str:
        if Path(src).suffix.lower() == ".json":
            return shutil.copy2(src, dst)
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=copy_json_or_link)


def inspect_glb(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(20)
        if len(header) < 20 or header[:4] != b"glTF":
            raise ValueError(f"not a GLB: {path}")
        _, version, declared_bytes, json_bytes, json_type = struct.unpack("<4sIIII", header)
        if json_type != 0x4E4F534A:
            raise ValueError(f"first GLB chunk is not JSON: {path}")
        document = json.loads(handle.read(json_bytes).decode("utf-8"))
    attributes: dict[str, int] = {}
    triangles = 0
    primitives = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitives += 1
            for semantic in primitive.get("attributes") or {}:
                attributes[semantic] = attributes.get(semantic, 0) + 1
            accessor_index = primitive.get("indices")
            if accessor_index is not None:
                triangles += int(document["accessors"][accessor_index]["count"]) // 3
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "version": version,
        "declaredBytes": declared_bytes,
        "meshes": len(document.get("meshes", [])),
        "primitives": primitives,
        "triangles": triangles,
        "images": len(document.get("images", [])),
        "textures": len(document.get("textures", [])),
        "materials": len(document.get("materials", [])),
        "attributes": attributes,
        "extensionsUsed": document.get("extensionsUsed") or [],
        "embeddedImages": sum("bufferView" in image for image in document.get("images", [])),
    }


def run(command: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip().splitlines()


def build_proxy(
    args: argparse.Namespace,
    row: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    shard_tileset = args.original_root / "detail" / "shards" / row["shard"] / "tileset.json"
    leaves = [(shard_tileset.parent / uri).resolve() for uri in row["originalLeafUris"]]
    missing = [str(path) for path in leaves if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing original leaves: {missing}")

    with tempfile.TemporaryDirectory(prefix=f"textured-proxy-{row['proxy']}-") as name:
        work = Path(name)
        merged = work / "merged.glb"
        simplified = work / "simplified.glb"
        merge_log = run([str(args.cli), "merge", *map(str, leaves), str(merged), "--merge-scenes"])
        merged_info = inspect_glb(merged)
        simplify_log = run(
            [
                str(args.cli),
                "simplify",
                str(merged),
                str(simplified),
                "--ratio",
                str(args.ratio),
                "--error",
                str(args.error),
                "--lock-border",
                args.lock_border,
                "--verbose",
            ]
        )
        final_info = inspect_glb(simplified)
        required = {"POSITION", "TEXCOORD_0"}
        if not required.issubset(final_info["attributes"]):
            raise ValueError(f"textured proxy lost required attributes: {final_info}")
        if final_info["images"] < 1 or final_info["textures"] < 1:
            raise ValueError(f"textured proxy contains no texture: {final_info}")
        if final_info["embeddedImages"] != final_info["images"]:
            raise ValueError(f"not all proxy images are embedded: {final_info}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.textured.tmp")
        shutil.copy2(simplified, temporary)
        os.replace(temporary, destination)
        final_info["path"] = str(destination)
    return {
        "proxy": row["proxy"],
        "shard": row["shard"],
        "screenFootprintRank": row["screenFootprintRank"],
        "descendantLeaves": len(leaves),
        "originalLeafGlbBytes": sum(path.stat().st_size for path in leaves),
        "merged": merged_info,
        "final": final_info,
        "mergeLog": merge_log[-5:],
        "simplifyLog": [line for line in simplify_log if "simplify:" in line][-10:],
    }


def main() -> None:
    args = parse_args()
    if args.selected_count < 1:
        raise ValueError("--selected-count must be positive")
    if args.selected_start_rank < 1:
        raise ValueError("--selected-start-rank must be positive")
    if not 0 < args.ratio <= 1 or not 0 < args.error <= 1:
        raise ValueError("ratio and error must be in (0, 1]")
    if not args.cli.is_file():
        raise FileNotFoundError(args.cli)

    lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
    ranked = sorted(lineage["visibleProxies"], key=lambda row: int(row["screenFootprintRank"]))
    selected = ranked[args.selected_start_rank - 1 : args.selected_start_rank - 1 + args.selected_count]
    if len(selected) != args.selected_count:
        raise ValueError(
            f"requested {args.selected_count} proxies from rank {args.selected_start_rank}, "
            f"but only found {len(selected)}"
        )
    source_before = json_hashes(args.candidate_root)
    clone_publication(args.candidate_root, args.output_root)

    proxies = []
    for row in selected:
        proxies.append(build_proxy(args, row, args.output_root / "proxy" / row["proxy"]))

    source_after = json_hashes(args.candidate_root)
    if source_before != source_after:
        raise RuntimeError("source publication JSON changed during POC build")
    report = {
        "candidateRoot": str(args.candidate_root),
        "originalRoot": str(args.original_root),
        "outputRoot": str(args.output_root),
        "lineage": str(args.lineage),
        "selectedCount": len(proxies),
        "selectedStartRank": args.selected_start_rank,
        "ratio": args.ratio,
        "error": args.error,
        "lockBorder": args.lock_border == "true",
        "sourcePublicationUnmodified": True,
        "totalFinalProxyBytes": sum(row["final"]["bytes"] for row in proxies),
        "totalFinalTriangles": sum(row["final"]["triangles"] for row in proxies),
        "totalEmbeddedImages": sum(row["final"]["embeddedImages"] for row in proxies),
        "proxies": proxies,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
