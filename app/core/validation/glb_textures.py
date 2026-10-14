#!/usr/bin/env python3
"""Audit embedded GLB textures without decoding them."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    version, declared = struct.unpack_from("<II", data, 4)
    if version != 2 or declared != len(data):
        raise ValueError(f"invalid GLB header: {path}")
    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \x00").decode("utf-8"))
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if document is None:
        raise ValueError(f"missing GLB JSON chunk: {path}")
    return document, binary


def ktx2_info(payload: bytes) -> dict[str, Any]:
    if len(payload) < 48 or payload[:12] != KTX2_MAGIC:
        return {"format": "unknown"}
    fields = struct.unpack_from("<9I", payload, 12)
    return {
        "format": "ktx2",
        "vkFormat": fields[0],
        "typeSize": fields[1],
        "width": fields[2],
        "height": fields[3],
        "depth": fields[4],
        "layers": fields[5],
        "faces": fields[6],
        "levels": fields[7],
        "supercompressionScheme": fields[8],
    }


def audit(path: Path) -> dict[str, Any]:
    document, binary = read_glb(path)
    rows = []
    hashes: defaultdict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(document.get("images", [])):
        if "bufferView" not in image:
            rows.append({"index": index, "embedded": False, "uri": image.get("uri")})
            continue
        view = document["bufferViews"][image["bufferView"]]
        start = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        payload = binary[start : start + length]
        digest = hashlib.sha256(payload).hexdigest()
        hashes[digest].append(index)
        rows.append(
            {
                "index": index,
                "embedded": True,
                "mimeType": image.get("mimeType"),
                "bytes": len(payload),
                "sha256": digest,
                **ktx2_info(payload),
            }
        )
    embedded = [row for row in rows if row.get("embedded")]
    dimensions = Counter(f"{row.get('width', 0)}x{row.get('height', 0)}" for row in embedded)
    duplicate_groups = [
        {"sha256": digest, "indices": indices, "count": len(indices)}
        for digest, indices in hashes.items()
        if len(indices) > 1
    ]
    return {
        "path": str(path),
        "glbBytes": path.stat().st_size,
        "images": len(rows),
        "embeddedImages": len(embedded),
        "embeddedImageBytes": sum(int(row["bytes"]) for row in embedded),
        "uniqueEmbeddedImages": len(hashes),
        "duplicateImageReferences": sum(len(group["indices"]) - 1 for group in duplicate_groups),
        "dimensions": dict(sorted(dimensions.items())),
        "largestImages": sorted(embedded, key=lambda row: int(row["bytes"]), reverse=True)[:20],
        "duplicateGroups": duplicate_groups,
        "textures": rows,
    }


def main() -> None:
    args = parse_args()
    reports = [audit(path) for path in args.inputs]
    all_hashes: Counter[str] = Counter(
        texture["sha256"]
        for report in reports
        for texture in report["textures"]
        if texture.get("embedded") and texture.get("sha256")
    )
    result = {
        "models": reports,
        "totalGlbBytes": sum(row["glbBytes"] for row in reports),
        "totalEmbeddedImageBytes": sum(row["embeddedImageBytes"] for row in reports),
        "totalImages": sum(row["images"] for row in reports),
        "totalUniqueWithinModels": sum(row["uniqueEmbeddedImages"] for row in reports),
        "totalDuplicateImageReferences": sum(row["duplicateImageReferences"] for row in reports),
        "uniqueImageHashesAcrossModels": len(all_hashes),
        "duplicateImageReferencesAcrossModels": sum(count - 1 for count in all_hashes.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
