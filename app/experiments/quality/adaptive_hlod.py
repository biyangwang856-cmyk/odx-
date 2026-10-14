#!/usr/bin/env python3
"""Validate adaptive HLOD report/file consistency and tile error monotonicity."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tileset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    document = json.loads(args.tileset.read_text())
    report = json.loads(args.report.read_text())
    shard_name = args.tileset.parent.name
    shard_report = next(item for item in report["shards"] if item["shard"] == shard_name)

    missing_content = []
    error_violations = []
    content_count = 0
    empty_internal_count = 0

    def visit(tile: dict, path: str) -> None:
        nonlocal content_count, empty_internal_count
        parent_error = float(tile.get("geometricError", 0.0))
        content = tile.get("content")
        if content:
            content_count += 1
            uri = content.get("uri") or content.get("url")
            if not uri or not (args.tileset.parent / uri).is_file():
                missing_content.append({"tile": path, "uri": uri})
        elif tile.get("children"):
            empty_internal_count += 1
        for index, child in enumerate(tile.get("children", [])):
            child_error = float(child.get("geometricError", 0.0))
            if parent_error + 1e-12 < child_error:
                error_violations.append(
                    {
                        "parent": path,
                        "child": f"{path}/{index}",
                        "parentError": parent_error,
                        "childError": child_error,
                    }
                )
            visit(child, f"{path}/{index}")

    visit(document["root"], "root")

    proxy_dir = args.report.parent / "proxy"
    accepted_missing = []
    rejected_present = []
    accepted = 0
    rejected = 0
    for node in shard_report["nodes"]:
        proxy = proxy_dir / node["proxy"]
        if node["accepted"]:
            accepted += 1
            if not proxy.is_file():
                accepted_missing.append(node["proxy"])
        else:
            rejected += 1
            if proxy.exists():
                rejected_present.append(node["proxy"])

    result = {
        "tileset": str(args.tileset),
        "report": str(args.report),
        "contentCount": content_count,
        "emptyInternalCount": empty_internal_count,
        "acceptedProxies": accepted,
        "rejectedCandidates": rejected,
        "missingContent": missing_content,
        "geometricErrorMonotonicityViolations": error_violations,
        "acceptedProxyFilesMissing": accepted_missing,
        "rejectedProxyFilesPresent": rejected_present,
    }
    result["passed"] = not any(
        (
            missing_content,
            error_violations,
            accepted_missing,
            rejected_present,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
