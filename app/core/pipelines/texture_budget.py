#!/usr/bin/env python3
"""Build an isolated candidate with a texture-resolution budget on selected proxies."""

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

from core.pipelines.textured_proxy import inspect_glb, json_hashes
from core.validation.glb_textures import audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--proxy", action="append", required=True)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--max-dimension", required=True, type=int)
    parser.add_argument("--quality", type=int, default=160)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def budget_proxy(
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

    with tempfile.TemporaryDirectory(prefix=f"texture-budget-{source.stem}-") as name:
        work = Path(name)
        decompressed = work / "decompressed.glb"
        resized = work / "resized.glb"
        recompressed = work / "recompressed.glb"
        logs = {
            "decompress": run([str(args.cli), "ktxdecompress", str(source), str(decompressed)]),
            "resize": run(
                [
                    str(args.cli),
                    "resize",
                    str(decompressed),
                    str(resized),
                    "--width",
                    str(args.max_dimension),
                    "--height",
                    str(args.max_dimension),
                ]
            ),
            "recompress": run(
                [
                    str(args.cli),
                    "etc1s",
                    str(resized),
                    str(recompressed),
                    "--quality",
                    str(args.quality),
                    "--jobs",
                    str(args.jobs),
                ]
            ),
        }
        after_model = inspect_glb(recompressed)
        after_textures = audit(recompressed)
        if before_model["triangles"] != after_model["triangles"]:
            raise ValueError("texture budgeting changed triangle count")
        if before_model["attributes"] != after_model["attributes"]:
            raise ValueError("texture budgeting changed vertex attributes")
        if before_model["images"] != after_model["images"]:
            raise ValueError("texture budgeting changed image count")
        if after_model["embeddedImages"] != after_model["images"]:
            raise ValueError("not all budgeted textures are embedded")

        temporary = destination.with_name(f".{destination.name}.texture-budget.tmp")
        shutil.copy2(recompressed, temporary)
        os.replace(temporary, destination)

    if sha256(source) != source_hash:
        raise RuntimeError(f"source proxy changed during build: {source}")
    after_model["path"] = str(destination)
    after_textures["path"] = str(destination)
    return {
        "proxy": str(relative),
        "sourceSha256": source_hash,
        "before": {"model": before_model, "textures": before_textures},
        "after": {"model": after_model, "textures": after_textures},
        "savedBytes": before_model["bytes"] - after_model["bytes"],
        "savedTextureBytes": before_textures["embeddedImageBytes"] - after_textures["embeddedImageBytes"],
        "logs": {stage: lines[-8:] for stage, lines in logs.items()},
    }


def main() -> None:
    args = parse_args()
    if args.max_dimension < 64:
        raise ValueError("--max-dimension must be at least 64")
    if not 1 <= args.quality <= 255:
        raise ValueError("--quality must be in [1, 255]")
    if not args.cli.is_file():
        raise FileNotFoundError(args.cli)

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")

    source_json_before = json_hashes(args.candidate_root)
    source_proxy_hashes = {proxy: sha256(args.candidate_root / proxy) for proxy in args.proxy}
    staging_root = args.output_root.with_name(f".{args.output_root.name}.building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(f"staging output already exists: {staging_root}")
    try:
        clone_publication(args.candidate_root, staging_root)
        proxies = [budget_proxy(args, Path(proxy), staging_root) for proxy in args.proxy]
        if json_hashes(args.candidate_root) != source_json_before:
            raise RuntimeError("source candidate JSON changed during texture budgeting")
        if any(
            sha256(args.candidate_root / proxy) != digest for proxy, digest in source_proxy_hashes.items()
        ):
            raise RuntimeError("source candidate proxy changed during texture budgeting")
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
        "maxDimension": args.max_dimension,
        "quality": args.quality,
        "jobs": args.jobs,
        "selectedCount": len(proxies),
        "sourceCandidateUnmodified": True,
        "savedBytes": sum(row["savedBytes"] for row in proxies),
        "savedTextureBytes": sum(row["savedTextureBytes"] for row in proxies),
        "proxies": proxies,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
