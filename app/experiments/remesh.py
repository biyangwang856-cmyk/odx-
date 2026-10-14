"""Research profile selection and command handling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from core.pipelines.remesh import RemeshProfile, build_publication

from experiments.context import TaskContext

PROFILES = {
    name: RemeshProfile(**values)
    for name, values in json.loads(
        (Path(__file__).parent / "specs/remesh-profiles.json").read_text(encoding="utf-8")
    ).items()
}


def _parser(profile_name: str, context: TaskContext) -> argparse.ArgumentParser:
    profile = PROFILES[profile_name]
    sibling_tool = (
        context.repo_root.parent / "third_party/3d-tiles/toolchain/node_modules/.bin/gltf-transform"
    )
    default_tool = (
        os.environ.get("GLTF_TRANSFORM")
        or shutil.which("gltf-transform")
        or (str(sibling_tool) if sibling_tool.is_file() else None)
    )
    parser = argparse.ArgumentParser(prog=f"just task build.remesh-{profile_name} --")
    parser.add_argument("--source", type=Path, default=context.data_root / profile.source)
    parser.add_argument("--output", type=Path, default=context.data_root / profile.output)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gltf-transform", default=default_tool)
    parser.add_argument("--node", default=os.environ.get("NODE") or shutil.which("node"))
    parser.add_argument("--basisu", default=os.environ.get("BASISU") or shutil.which("basisu"))
    parser.add_argument(
        "--sharp-module",
        default=os.environ.get("SHARP_MODULE"),
        help="Path to the locked Sharp CommonJS module used to decode source textures.",
    )
    parser.add_argument(
        "--basisu-provenance",
        default=os.environ.get("BASISU_PROVENANCE"),
        help="Locked Basis Universal source provenance JSON.",
    )
    return parser


def run_profile(profile_name: str, context: TaskContext, arguments: Sequence[str]) -> int:
    """Parse profile overrides, render dry-run data, or build the publication."""
    profile = PROFILES[profile_name]
    args = _parser(profile_name, context).parse_args(arguments)
    configuration = {
        "profile": asdict(profile),
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "workers": args.workers,
        "gltfTransform": args.gltf_transform or "<required:gltf-transform>",
        "node": args.node or "<required:node>",
        "basisu": args.basisu or "<required:basisu-1.16.3>",
        "sharpModule": args.sharp_module or "<required:sharp-module>",
        "basisuProvenance": args.basisu_provenance or "<required:basisu-provenance>",
    }
    if context.dry_run:
        print(json.dumps(configuration, indent=2, ensure_ascii=False))
        return 0
    missing = [
        name
        for name, value in (
            ("gltf-transform", args.gltf_transform),
            ("node", args.node),
            ("basisu", args.basisu),
            ("sharp-module", args.sharp_module),
            ("basisu-provenance", args.basisu_provenance),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"required v55 tools are not configured: {', '.join(missing)}")
    report = build_publication(
        profile=profile,
        source=args.source.resolve(),
        output=args.output.resolve(),
        executable=args.gltf_transform,
        workers=args.workers,
        node_executable=args.node,
        basisu_executable=args.basisu,
        sharp_module=args.sharp_module,
        basisu_provenance=args.basisu_provenance,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
