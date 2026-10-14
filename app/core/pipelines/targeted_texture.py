#!/usr/bin/env python3
"""Build a non-destructive texture-camera-targeted HLOD schedule candidate.

The output hard-links all unchanged assets from the source publication and
rewrites only tileset JSON.  Selected proxy subtrees inherit the original V17
geometric-error schedule; all other subtrees retain the v6 P99 schedule.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

PROXY_RE = re.compile(r"mesh-(\d{2})-(\d{3})\.glb$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--restore-content",
        action="store_true",
        help="also restore original content/refine fields inside selected subtrees",
    )
    parser.add_argument(
        "--selected-count",
        type=int,
        help="override lineage selection with the first N screen-footprint-ranked proxies",
    )
    parser.add_argument(
        "--selected-root-additive-fallback",
        action="store_true",
        help="keep each selected root proxy with ADD refinement while restoring descendants",
    )
    parser.add_argument(
        "--selected-root-error-m",
        type=float,
        help="minimum geometric error assigned to selected roots before ancestor propagation",
    )
    return parser.parse_args()


def content_uri(tile: dict[str, Any]) -> str | None:
    content = tile.get("content") or {}
    return content.get("uri") or content.get("url")


def clone_publication(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    def hardlink_or_copy(src: str, dst: str) -> str:
        if Path(src).suffix.lower() == ".json":
            return shutil.copy2(src, dst)
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=hardlink_or_copy)


def json_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*.json")):
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def rewrite_shard(
    candidate_tileset: Path,
    original_tileset: Path,
    output_tileset: Path,
    selected: set[str],
    restore_content: bool,
    selected_root_additive_fallback: bool,
    selected_root_error_m: float | None,
) -> dict[str, Any]:
    candidate = json.loads(candidate_tileset.read_text(encoding="utf-8"))
    original = json.loads(original_tileset.read_text(encoding="utf-8"))
    replacements: list[dict[str, Any]] = []
    ancestor_raises: list[dict[str, Any]] = []
    content_changes: list[dict[str, Any]] = []

    def visit(
        c_tile: dict[str, Any],
        o_tile: dict[str, Any],
        path: tuple[int, ...],
        restore_original: bool,
    ) -> float:
        c_children = c_tile.get("children") or []
        o_children = o_tile.get("children") or []
        if len(c_children) != len(o_children):
            raise ValueError(
                f"tree mismatch at {'.'.join(map(str, path)) or 'root'}: "
                f"candidate={len(c_children)}, original={len(o_children)}"
            )

        uri = content_uri(c_tile)
        proxy_name = Path(uri).name if uri and PROXY_RE.search(Path(uri).name) else None
        selected_here = bool(proxy_name and proxy_name in selected)
        restore_here = restore_original or selected_here
        old_error = float(c_tile.get("geometricError") or 0.0)
        candidate_content_before = copy.deepcopy(c_tile.get("content"))

        child_errors = [
            visit(c_child, o_child, path + (index,), restore_here)
            for index, (c_child, o_child) in enumerate(zip(c_children, o_children))
        ]

        if restore_here:
            new_error = float(o_tile.get("geometricError") or 0.0)
            if child_errors:
                new_error = max(new_error, max(child_errors))
            if selected_here and selected_root_error_m is not None:
                new_error = max(new_error, selected_root_error_m)
            c_tile["geometricError"] = new_error
            if restore_content:
                old_uri = content_uri(c_tile)
                original_content = o_tile.get("content")
                if original_content is None:
                    c_tile.pop("content", None)
                else:
                    c_tile["content"] = copy.deepcopy(original_content)
                if "refine" in o_tile:
                    c_tile["refine"] = o_tile["refine"]
                else:
                    c_tile.pop("refine", None)
                if selected_here and selected_root_additive_fallback and candidate_content_before:
                    c_tile["content"] = candidate_content_before
                    c_tile["refine"] = "ADD"
                new_uri = content_uri(c_tile)
                if old_uri != new_uri:
                    content_changes.append(
                        {
                            "path": list(path),
                            "oldContent": old_uri,
                            "newContent": new_uri,
                        }
                    )
            replacements.append(
                {
                    "path": list(path),
                    "proxy": proxy_name,
                    "selectedRoot": selected_here,
                    "oldGeometricErrorM": old_error,
                    "newGeometricErrorM": new_error,
                }
            )
            return new_error

        # A selected subtree can sit below an unchanged proxy.  Raise only the
        # ancestor error needed to keep that branch traversable.
        new_error = max([old_error, *child_errors]) if child_errors else old_error
        if new_error > old_error + 1e-12:
            c_tile["geometricError"] = new_error
            ancestor_raises.append(
                {
                    "path": list(path),
                    "proxy": proxy_name,
                    "oldGeometricErrorM": old_error,
                    "newGeometricErrorM": new_error,
                }
            )
        return new_error

    root_error = visit(candidate["root"], original["root"], (), False)
    output_tileset.write_text(
        json.dumps(candidate, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    selected_seen = sorted(
        row["proxy"] for row in replacements if row.get("selectedRoot") and row.get("proxy")
    )
    missing = sorted(selected - set(selected_seen))
    if missing:
        raise KeyError(f"selected proxies missing from shard tree: {missing}")
    return {
        "rootGeometricErrorM": root_error,
        "selectedRoots": selected_seen,
        "restoredNodes": len(replacements),
        "raisedAncestors": len(ancestor_raises),
        "changedContents": len(content_changes),
        "maxRestoredErrorM": max((row["newGeometricErrorM"] for row in replacements), default=0.0),
        "replacements": replacements,
        "ancestorRaises": ancestor_raises,
        "contentChanges": content_changes,
        "selectedRootAdditiveFallback": selected_root_additive_fallback,
        "selectedRootErrorM": selected_root_error_m,
    }


def count_inodes(root: Path) -> dict[str, int]:
    files = 0
    json_files = 0
    linked_to_source = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        json_files += int(path.suffix.lower() == ".json")
        try:
            linked_to_source += int(path.stat().st_nlink > 1)
        except OSError:
            pass
    return {"files": files, "jsonFiles": json_files, "filesWithMultipleLinks": linked_to_source}


def main() -> None:
    args = parse_args()
    lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
    ranked_rows = sorted(
        lineage["visibleProxies"], key=lambda row: int(row.get("screenFootprintRank") or 10**9)
    )
    if args.selected_count is not None:
        if args.selected_count < 1:
            raise ValueError("--selected-count must be positive")
        selected_rows = ranked_rows[: args.selected_count]
    else:
        selected_rows = [row for row in ranked_rows if row.get("selectedForTargetedV7")]
    by_shard: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        by_shard[row["shard"]].add(row["proxy"])
    if not by_shard:
        raise ValueError("lineage report contains no selected proxies")

    source_json_before = json_hashes(args.candidate_root)
    clone_publication(args.candidate_root, args.output_root)
    shard_reports: dict[str, Any] = {}
    try:
        for shard, selected in sorted(by_shard.items()):
            relative = Path("detail") / "shards" / shard / "tileset.json"
            shard_reports[shard] = rewrite_shard(
                args.candidate_root / relative,
                args.original_root / relative,
                args.output_root / relative,
                selected,
                args.restore_content,
                args.selected_root_additive_fallback,
                args.selected_root_error_m,
            )
    except Exception:
        # Preserve the failed output for forensic inspection; never delete a
        # large publication automatically after a partial build.
        raise

    source_json_after = json_hashes(args.candidate_root)
    if source_json_before != source_json_after:
        raise RuntimeError("source publication JSON changed while building candidate")
    report = {
        "candidateRoot": str(args.candidate_root),
        "originalRoot": str(args.original_root),
        "outputRoot": str(args.output_root),
        "lineage": str(args.lineage),
        "policy": (
            "retain v6 everywhere; restore original V17 geometric errors"
            + (" and content structure" if args.restore_content else "")
            + " only inside selected fixed-camera proxy subtrees and raise their ancestors"
        ),
        "restoreContent": args.restore_content,
        "selectedRootAdditiveFallback": args.selected_root_additive_fallback,
        "selectedRootErrorM": args.selected_root_error_m,
        "selectedProxyCount": len(selected_rows),
        "selectedByShard": {shard: sorted(names) for shard, names in sorted(by_shard.items())},
        "selectedDescendantLeaves": sum(int(row.get("descendantLeaves") or 0) for row in selected_rows),
        "shards": shard_reports,
        "restoredNodes": sum(row["restoredNodes"] for row in shard_reports.values()),
        "raisedAncestors": sum(row["raisedAncestors"] for row in shard_reports.values()),
        "changedContents": sum(row["changedContents"] for row in shard_reports.values()),
        "fileInventory": count_inodes(args.output_root),
        "sourcePublicationUnmodified": source_json_before == source_json_after,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "shards"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
