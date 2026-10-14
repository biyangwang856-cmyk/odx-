#!/usr/bin/env python3
"""Reconstruct a v6 shard schedule exactly from its immutable build report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--shards", required=True, nargs="+")
    parser.add_argument("--backup-suffix", default=".pre-p99-repair-20260824.json")
    return parser.parse_args()


def atomic_json_write(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def restore_shard(publication: Path, shard: str, backup_suffix: str) -> dict[str, Any]:
    tileset_path = publication / "detail" / "shards" / shard / "tileset.json"
    report_path = publication / f"intermediate-mesh-hlod-shard-{shard}-report.json"
    document = json.loads(tileset_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reports = report["nodes"]
    report_index = 0
    accepted = 0
    rejected = 0
    skipped = 0
    leaves = 0

    def visit(tile: dict[str, Any]) -> tuple[int, float]:
        nonlocal report_index, accepted, rejected, skipped, leaves
        children = tile.get("children") or []
        if not children:
            tile["geometricError"] = 0.0
            tile["refine"] = "REPLACE"
            if tile.get("content"):
                leaves += 1
                return 1, 0.0
            return 0, 0.0

        descendant_leaves = 0
        child_error = 0.0
        for child in children:
            count, error = visit(child)
            descendant_leaves += count
            child_error = max(child_error, error)
        if descendant_leaves < 2:
            skipped += 1
            tile.pop("content", None)
            tile["geometricError"] = child_error
            tile["refine"] = "REPLACE"
            return descendant_leaves, child_error

        if report_index >= len(reports):
            raise IndexError(f"report exhausted in shard {shard}")
        row = reports[report_index]
        expected = f"mesh-{shard}-{report_index:03d}.glb"
        report_index += 1
        if row["proxy"] != expected:
            raise ValueError(f"report order mismatch: expected {expected}, got {row['proxy']}")
        if int(row["descendantLeaves"]) != descendant_leaves:
            raise ValueError(
                f"leaf count mismatch for {expected}: tree={descendant_leaves}, "
                f"report={row['descendantLeaves']}"
            )
        if row["accepted"]:
            accepted += 1
            tile["content"] = {"uri": f"../../../proxy/{expected}"}
        else:
            rejected += 1
            tile.pop("content", None)
        error = float(row["geometricErrorM"])
        tile["geometricError"] = error
        tile["refine"] = "REPLACE"
        return descendant_leaves, error

    _, root_error = visit(document["root"])
    if report_index != len(reports):
        raise ValueError(f"unused report rows in shard {shard}: {len(reports) - report_index}")
    document["geometricError"] = root_error

    backup = tileset_path.with_name(tileset_path.name + backup_suffix)
    if not backup.exists():
        shutil.copy2(tileset_path, backup)
    atomic_json_write(tileset_path, document)
    return {
        "shard": shard,
        "tileset": str(tileset_path),
        "backup": str(backup),
        "rootGeometricErrorM": root_error,
        "reportRows": report_index,
        "accepted": accepted,
        "rejected": rejected,
        "skippedInternalNodes": skipped,
        "leafContents": leaves,
        "matchesReport": (
            accepted == int(report["proxies"])
            and rejected == int(report["rejectedCandidates"])
            and skipped == int(report["skippedInternalNodes"])
            and leaves == int(report["leafContents"])
            and abs(root_error - float(report["rootGeometricError"])) < 1e-12
        ),
    }


def main() -> None:
    args = parse_args()
    results = [restore_shard(args.publication, shard.zfill(2), args.backup_suffix) for shard in args.shards]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if not all(result["matchesReport"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
