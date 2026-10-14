#!/usr/bin/env python3
"""Clone an adaptive HLOD publication and recompute only LOD scheduling errors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("p99", "maximum", "original"), default="maximum")
    parser.add_argument("--safety-factor", type=float, default=1.0)
    parser.add_argument("--cap-m", type=float)
    parser.add_argument("--original-root", type=Path)
    return parser.parse_args()


def copy_publication(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    def copy_file(src: str, dst: str) -> str:
        if Path(src).suffix.lower() == ".json":
            return shutil.copy2(src, dst)
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=copy_file)


def apply_error_policy(value: float, safety_factor: float, cap_m: float | None) -> float:
    value *= safety_factor
    if cap_m is not None:
        value = min(value, cap_m)
    return float(value)


def metric_value(node: dict, metric: str, safety_factor: float, cap_m: float | None) -> float:
    measured = node["measuredError"]
    value = measured["maximumM"] if metric == "maximum" else measured["areaWeightedP99M"]
    return apply_error_policy(value, safety_factor, cap_m)


def reschedule_shard(
    shard_path: Path,
    report_path: Path,
    metric: str,
    safety_factor: float,
    cap_m: float | None,
    original_path: Path | None = None,
) -> dict:
    tileset = json.loads(shard_path.read_text())
    original_tileset = json.loads(original_path.read_text()) if original_path else None
    report = json.loads(report_path.read_text())
    accepted_proxies = {node["proxy"] for node in report["nodes"] if node["accepted"]}
    proxy_errors = {
        node["proxy"]: metric_value(node, metric, safety_factor, cap_m)
        for node in report["nodes"]
        if node["accepted"] and metric != "original"
    }
    scheduled = []

    def visit(tile: dict, original_tile: dict | None = None) -> float:
        children = tile.get("children", [])
        original_children = original_tile.get("children", []) if original_tile else []
        if original_tile is not None and len(children) != len(original_children):
            raise ValueError(f"Candidate/original tree mismatch in shard {report['shard']}")
        child_error = max(
            (
                visit(child, original_children[index] if original_tile else None)
                for index, child in enumerate(children)
            ),
            default=0.0,
        )
        own_error = 0.0
        original_error = float(original_tile.get("geometricError", 0.0)) if original_tile else 0.0
        content = tile.get("content")
        if content:
            uri = content.get("uri", content.get("url", ""))
            proxy_name = Path(uri).name
            if proxy_name in accepted_proxies:
                own_error = (
                    apply_error_policy(original_error, safety_factor, cap_m)
                    if metric == "original"
                    else proxy_errors[proxy_name]
                )
                scheduled.append({"proxy": proxy_name, "ownErrorM": own_error})
        error = max(own_error, child_error, original_error)
        tile["geometricError"] = error
        tile["refine"] = "REPLACE"
        return error

    root_error = visit(tileset["root"], original_tileset["root"] if original_tileset else None)
    tileset["geometricError"] = root_error
    shard_path.write_text(json.dumps(tileset, indent=2) + "\n")
    return {
        "shard": report["shard"],
        "rootGeometricError": root_error,
        "scheduledProxies": len(scheduled),
        "minimumOwnErrorM": min((item["ownErrorM"] for item in scheduled), default=0.0),
        "maximumOwnErrorM": max((item["ownErrorM"] for item in scheduled), default=0.0),
    }


def main() -> None:
    args = parse_args()
    if args.safety_factor <= 0:
        raise ValueError("safety-factor must be positive")
    if args.cap_m is not None and args.cap_m <= 0:
        raise ValueError("cap-m must be positive")
    if args.metric == "original" and args.original_root is None:
        raise ValueError("--original-root is required for metric=original")
    copy_publication(args.input, args.output)
    shards = []
    for shard_path in sorted((args.output / "detail" / "shards").glob("*/tileset.json")):
        shard = shard_path.parent.name
        report_path = args.output / f"intermediate-mesh-hlod-shard-{shard}-report.json"
        original_path = (
            args.original_root / "detail" / "shards" / shard / "tileset.json"
            if args.metric == "original"
            else None
        )
        shards.append(
            reschedule_shard(
                shard_path,
                report_path,
                args.metric,
                args.safety_factor,
                args.cap_m,
                original_path,
            )
        )
    result = {
        "input": str(args.input),
        "output": str(args.output),
        "metric": args.metric,
        "safetyFactor": args.safety_factor,
        "capM": args.cap_m,
        "originalRoot": str(args.original_root) if args.original_root else None,
        "shards": shards,
    }
    report_path = args.output / "intermediate-mesh-hlod-reschedule-report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
