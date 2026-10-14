"""Configurable Remesh publication profiles shared by registered build tasks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.validation.glb_textures import read_glb


@dataclass(frozen=True)
class RemeshProfile:
    """Parameters for the clean-break v55 production publication."""

    name: str
    source: str
    output: str
    simplify_ratio: float | None
    simplify_error: float | None
    webp_quality: int | None
    webp_effort: int | None
    meshopt_level: str | None
    gpu_texture_codec: str
    gpu_texture_quality: int
    gpu_texture_compression_level: int
    gpu_texture_minimum_psnr: float
    gpu_texture_maximum_byte_ratio: float


def audit_publication(root: Path) -> dict[str, Any]:
    """Summarize GLB geometry, image payload, and compression extensions."""
    result: dict[str, Any] = {
        "tiles": 0,
        "bytes": 0,
        "triangles": 0,
        "images": 0,
        "imageBytes": 0,
        "meshoptTiles": 0,
        "imageFormats": Counter(),
    }
    for path in sorted(root.rglob("*.glb")):
        document, _ = read_glb(path)
        result["tiles"] += 1
        result["bytes"] += path.stat().st_size
        if "EXT_meshopt_compression" in document.get("extensionsUsed", []):
            result["meshoptTiles"] += 1
        buffer_views = document.get("bufferViews", [])
        for image in document.get("images", []):
            result["images"] += 1
            result["imageFormats"][image.get("mimeType", "unknown")] += 1
            if "bufferView" in image:
                result["imageBytes"] += int(buffer_views[image["bufferView"]]["byteLength"])
        accessors = document.get("accessors", [])
        for mesh in document.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                if "indices" in primitive:
                    result["triangles"] += int(accessors[primitive["indices"]]["count"]) // 3
    result["imageFormats"] = dict(sorted(result["imageFormats"].items()))
    return result


def _run_transform(command: Sequence[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(f"gltf-transform failed ({completed.returncode}): {output[-2000:]}")


def _transform_tile(
    source: Path,
    destination: Path,
    work: Path,
    profile: RemeshProfile,
    executable: str,
    *,
    node_executable: str,
    basisu_executable: str,
    sharp_module: str,
    basisu_provenance: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(os.fsencode(source)).hexdigest()[:20]
    current = source
    temporaries: list[Path] = []

    if profile.simplify_ratio is not None:
        simplified = work / f"{token}.simplified.glb"
        _run_transform(
            [
                executable,
                "simplify",
                str(current),
                str(simplified),
                "--ratio",
                str(profile.simplify_ratio),
                "--error",
                str(profile.simplify_error),
                "--lock-border",
            ]
        )
        current = simplified
        temporaries.append(simplified)

    if profile.webp_quality is not None:
        webp = work / f"{token}.webp.glb"
        _run_transform(
            [
                executable,
                "webp",
                str(current),
                str(webp),
                "--quality",
                str(profile.webp_quality),
                "--effort",
                str(profile.webp_effort),
            ]
        )
        current = webp
        temporaries.append(webp)

    mesh_output = destination
    if profile.meshopt_level is not None:
        mesh_output = work / f"{token}.meshopt.glb"
        _run_transform(
            [
                executable,
                "meshopt",
                str(current),
                str(mesh_output),
                "--level",
                profile.meshopt_level,
            ]
        )
        temporaries.append(mesh_output)
    elif current != destination:
        shutil.copy2(current, mesh_output)

    policy = json.dumps(
        {
            "codec": profile.gpu_texture_codec,
            "quality": profile.gpu_texture_quality,
            "compressionLevel": profile.gpu_texture_compression_level,
            "minimumBc7RgbPsnr": profile.gpu_texture_minimum_psnr,
            "maximumByteRatio": profile.gpu_texture_maximum_byte_ratio,
        },
        separators=(",", ":"),
    )
    optimizer = Path(__file__).with_name("gpu_textures.cjs")
    completed = subprocess.run(
        [
            node_executable,
            str(optimizer),
            str(mesh_output),
            str(destination),
            basisu_executable,
            sharp_module,
            basisu_provenance,
            policy,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(f"GPU texture optimization failed ({completed.returncode}): {detail[-2000:]}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("GPU texture optimizer returned an invalid report") from error

    for temporary in temporaries:
        temporary.unlink(missing_ok=True)
    return report


def build_publication(
    *,
    profile: RemeshProfile,
    source: Path,
    output: Path,
    executable: str,
    workers: int,
    node_executable: str,
    basisu_executable: str,
    sharp_module: str,
    basisu_provenance: str,
) -> dict[str, Any]:
    """Build one publication into a new destination and atomically reveal it."""
    if not source.is_dir() or not (source / "tileset.json").is_file():
        raise FileNotFoundError(f"source publication is missing tileset.json: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new versioned path: {output}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / "publication"
        work = Path(temporary) / "work"
        staging.mkdir()
        work.mkdir()
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".glb":
                destination = staging / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

        glbs = sorted(source.rglob("*.glb"))

        def transform(path: Path) -> dict[str, Any]:
            report = _transform_tile(
                path,
                staging / path.relative_to(source),
                work,
                profile,
                executable,
                node_executable=node_executable,
                basisu_executable=basisu_executable,
                sharp_module=sharp_module,
                basisu_provenance=basisu_provenance,
            )
            return {"path": path.relative_to(source).as_posix(), **report}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            texture_reports = list(executor.map(transform, glbs))

        source_report = audit_publication(source)
        output_report = audit_publication(staging)
        report = {
            "name": profile.name,
            "profile": asdict(profile),
            "source": source_report,
            "output": output_report,
            "byteRatio": (
                output_report["bytes"] / source_report["bytes"] if source_report["bytes"] else None
            ),
            "triangleRatio": (
                output_report["triangles"] / source_report["triangles"]
                if source_report["triangles"]
                else None
            ),
            "gpuTextures": {
                "codec": profile.gpu_texture_codec,
                "processedModels": len(texture_reports),
                "modifiedModels": sum(row["status"] == "converted" for row in texture_reports),
                "eligibleImages": sum(row["eligibleImages"] for row in texture_reports),
                "convertedImages": sum(row["convertedImages"] for row in texture_reports),
                "geometryAndUvBytesPreserved": all(
                    row["geometryAndUvBytesPreserved"] for row in texture_reports
                ),
                "models": texture_reports,
            },
        }
        report_payload = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        report["reportPayloadDigest"] = f"sha256:{hashlib.sha256(report_payload).hexdigest()}"
        (staging / "build-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    return report
