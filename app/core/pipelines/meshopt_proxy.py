#!/usr/bin/env python3
"""Build an isolated publication with Meshopt-compressed selected proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.pipelines.textured_proxy import clone_publication, inspect_glb, json_hashes
from core.validation.glb_textures import audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--proxy", action="append", required=True)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--level", choices=("medium", "high"), default="high")
    parser.add_argument("--quantize-position", type=int, default=14)
    parser.add_argument("--quantize-texcoord", type=int, default=12)
    parser.add_argument("--quantize-normal", type=int, default=10)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {rendered}\n{completed.stdout.rstrip()}"
        )
    return completed.stdout.strip().splitlines()


def texture_hashes(info: dict[str, Any]) -> list[str]:
    return sorted(row["sha256"] for row in info["textures"] if row.get("embedded") and row.get("sha256"))


def compress_proxy(
    args: argparse.Namespace,
    relative: Path,
    output_root: Path,
) -> dict[str, Any]:
    source = args.candidate_root / relative
    destination = output_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)

    source_hash = sha256(source)
    before_model = inspect_glb(source)
    before_textures = audit(source)

    with tempfile.TemporaryDirectory(prefix=f"meshopt-{source.stem}-") as name:
        compressed = Path(name) / "compressed.glb"
        log = run(
            [
                str(args.cli),
                "meshopt",
                str(source),
                str(compressed),
                "--level",
                args.level,
                "--quantize-position",
                str(args.quantize_position),
                "--quantize-texcoord",
                str(args.quantize_texcoord),
                "--quantize-normal",
                str(args.quantize_normal),
                "--verbose",
            ]
        )
        after_model = inspect_glb(compressed)
        after_textures = audit(compressed)

        if before_model["triangles"] != after_model["triangles"]:
            raise ValueError("Meshopt changed triangle count")
        if before_model["attributes"] != after_model["attributes"]:
            raise ValueError("Meshopt changed vertex attribute coverage")
        if before_model["images"] != after_model["images"]:
            raise ValueError("Meshopt changed image count")
        if before_model["materials"] != after_model["materials"]:
            raise ValueError("Meshopt changed material count")
        if texture_hashes(before_textures) != texture_hashes(after_textures):
            raise ValueError("Meshopt changed embedded texture payloads")
        if "EXT_meshopt_compression" not in after_model["extensionsUsed"]:
            raise ValueError("Meshopt output does not declare EXT_meshopt_compression")

        temporary = destination.with_name(f".{destination.name}.meshopt.tmp")
        shutil.copy2(compressed, temporary)
        os.replace(temporary, destination)

    if sha256(source) != source_hash:
        raise RuntimeError(f"source proxy changed during Meshopt build: {source}")
    after_model["path"] = str(destination)
    after_textures["path"] = str(destination)
    return {
        "proxy": str(relative),
        "sourceSha256": source_hash,
        "before": {"model": before_model, "textures": before_textures},
        "after": {"model": after_model, "textures": after_textures},
        "savedBytes": before_model["bytes"] - after_model["bytes"],
        "texturePayloadUnchanged": True,
        "log": log[-12:],
    }


def main() -> None:
    args = parse_args()
    if not args.cli.is_file():
        raise FileNotFoundError(args.cli)
    for value, name in (
        (args.quantize_position, "--quantize-position"),
        (args.quantize_texcoord, "--quantize-texcoord"),
        (args.quantize_normal, "--quantize-normal"),
    ):
        if not 1 <= value <= 16:
            raise ValueError(f"{name} must be in [1, 16]")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")

    source_json_before = json_hashes(args.candidate_root)
    source_proxy_hashes = {proxy: sha256(args.candidate_root / proxy) for proxy in args.proxy}
    staging_root = args.output_root.with_name(f".{args.output_root.name}.building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(f"staging output already exists: {staging_root}")

    try:
        clone_publication(args.candidate_root, staging_root)
        proxies = [compress_proxy(args, Path(proxy), staging_root) for proxy in args.proxy]
        if json_hashes(args.candidate_root) != source_json_before:
            raise RuntimeError("source candidate JSON changed during Meshopt build")
        if any(
            sha256(args.candidate_root / proxy) != digest for proxy, digest in source_proxy_hashes.items()
        ):
            raise RuntimeError("source candidate proxy changed during Meshopt build")
        os.replace(staging_root, args.output_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    for row in proxies:
        relative = Path(row["proxy"])
        row["after"]["model"]["path"] = str(args.output_root / relative)
        row["after"]["textures"]["path"] = str(args.output_root / relative)

    report = {
        "candidateRoot": str(args.candidate_root),
        "outputRoot": str(args.output_root),
        "level": args.level,
        "quantization": {
            "position": args.quantize_position,
            "texcoord": args.quantize_texcoord,
            "normal": args.quantize_normal,
        },
        "selectedCount": len(proxies),
        "sourceCandidateUnmodified": True,
        "totalSavedBytes": sum(row["savedBytes"] for row in proxies),
        "proxies": proxies,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
