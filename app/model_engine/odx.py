"""One-shot adapter for digest-pinned ODX reconstruction."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.artifacts import artifact_ref, image_tree_identity, safe_extract_tar
from core.contracts._validation import canonical_json_digest
from core.contracts.odx_output import odx_checkpoint_binding, validate_odx_output_manifest
from core.contracts.photogrammetry_odx import (
    photogrammetry_odx_input_digest,
    validate_photogrammetry_odx_input,
)
from core.validation.odx_output import output_files, read_origin_offset, validate_cog, validate_textured_mesh

from model_engine.common import AdapterEnvironment, EventEmitter, event_artifact_ref

ODX_IMAGE = (
    "docker.io/webodm/odx@"
    "sha256:f67dac749fa33009ae4f42bcf84e36f14b39bf5f7d280d838f87b8b6128c14a5"
)
ODX_SOURCE_REVISION = "eb9fd56ac430fc3735401d88aaef9945ebd00e14"
PARAMETERS = {
    "featureQuality": "--feature-quality",
    "matcherNeighbors": "--matcher-neighbors",
    "pcQuality": "--pc-quality",
    "meshSize": "--mesh-size",
    "meshOctreeDepth": "--mesh-octree-depth",
}


def build_odx_command(dataset_root: Path, manifest: Mapping[str, Any]) -> list[str]:
    validate_photogrammetry_odx_input(manifest)
    if manifest["odxParameters"].get("texturingToneMapping") not in (None, "none"):
        raise ValueError("ODX 3.8.3 does not support texturingToneMapping")
    if {"groundControl", "geoFile", "imageGroups"} & manifest.keys():
        raise ValueError("this adapter does not support optional control-file inputs")
    command = [
        sys.executable,
        "/code/run.py",
        "--project-path",
        str(dataset_root.resolve()),
        "--end-with",
        "odm_orthophoto",
        "--cog",
        "--matcher-type",
        "flann",
        "--sfm-algorithm",
        "incremental",
        "--orthophoto-compression",
        "DEFLATE",
        "--max-concurrency",
        str(min(manifest["resourceBudget"]["maxCpuCores"], 32)),
        "project",
    ]
    # Both modes retain the complete 3D model for downstream viewers.
    # ODX builds an additional nadir-textured 2.5D surface when this flag is absent.
    if manifest["odxParameters"].get("orthophotoMode", "3d") == "3d":
        command.append("--use-3dmesh")
    for name, flag in PARAMETERS.items():
        if name in manifest["odxParameters"]:
            command.extend([flag, str(manifest["odxParameters"][name]).lower()])
    return command


def build_odx_output_manifest(output_root: Path, input_manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_photogrammetry_odx_input(input_manifest)
    obj = output_root / "odm_texturing/odm_textured_model_geo.obj"
    mtl = output_root / "odm_texturing/odm_textured_model_geo.mtl"
    orthophoto = output_root / "odm_orthophoto/odm_orthophoto.tif"
    for path in (obj, mtl, orthophoto):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ODX output is missing or a symlink: {path.name}")
    validate_cog(orthophoto, input_manifest["crs"]["horizontal"])
    offset = read_origin_offset(
        output_root / "odm_georeferencing/coords.txt", input_manifest["crs"]["horizontal"]
    )
    textures = validate_textured_mesh(obj, mtl)
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    if any(path.suffix.lower() not in media for path in textures):
        raise ValueError("ODX texture format is unsupported")
    artifacts = {
        "orthophoto": artifact_ref(orthophoto, output_root, "image/tiff", "odx-orthophoto"),
        "mesh": {
            "obj": artifact_ref(obj, output_root, "model/obj", "odx-obj"),
            "mtl": artifact_ref(mtl, output_root, "model/mtl", "odx-mtl"),
            "textures": [
                artifact_ref(path, output_root, media[path.suffix.lower()], "odx-texture")
                for path in textures
            ],
        },
    }
    coordinates = {**input_manifest["crs"], "originOffsetMeters": offset}
    producer = {
        "engine": "odx",
        "engineVersion": "3.8.3",
        "imageDigest": ODX_IMAGE.split("@", 1)[1],
        "sourceRevision": ODX_SOURCE_REVISION,
    }
    binding = odx_checkpoint_binding(
        {"producer": producer, "coordinates": coordinates, "artifacts": artifacts},
        input_manifest,
    )
    manifest = {
        "schemaVersion": "odx-output-manifest.v2",
        "outputId": f"odx-output:{binding[-24:]}",
        "input": {
            "manifestDigest": photogrammetry_odx_input_digest(input_manifest),
            "inputId": input_manifest["inputId"],
            "imageContentDigest": input_manifest["images"]["contentDigest"],
        },
        "producer": producer,
        "parametersDigest": canonical_json_digest(input_manifest["odxParameters"]),
        "checkpoint": {
            "checkpointId": f"checkpoint:odx:{binding[-24:]}",
            "stage": "odm_orthophoto",
            "contentDigest": binding,
        },
        "coordinates": coordinates,
        "artifacts": artifacts,
    }
    validate_odx_output_manifest(manifest, input_manifest)
    return manifest


def _verify_images(root: Path, manifest: Mapping[str, Any]) -> None:
    observed = image_tree_identity(root)
    expected = manifest["images"]
    if (
        observed["algorithm"] != expected["treeAlgorithm"]
        or observed["digest"] != expected["contentDigest"]
        or observed["count"] != expected["fileCount"]
        or observed["bytes"] != expected["bytes"]
    ):
        raise ValueError("image tree does not match photogrammetry.odx.v1 input identity")


def create_odx_bundle(
    project: Path,
    input_manifest: Path,
    output_manifest: Path,
    bundle_path: Path,
) -> None:
    with tarfile.open(bundle_path, "w") as bundle:
        manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
        for artifact in output_files(manifest):
            relative = artifact["relativePath"]
            bundle.add(project / relative, arcname=relative, recursive=False)
        bundle.add(input_manifest, arcname="input-manifest.json")
        bundle.add(output_manifest, arcname="odx-output-manifest.json")


def classify_adapter_error(error: BaseException, stage: str) -> tuple[str, int]:
    if stage == "startup":
        return "contract_mismatch", 2
    if stage == "output":
        return "artifact_invalid", 2
    if isinstance(error, (FileNotFoundError, ValueError)):
        return "invalid_input", 2
    return "adapter_failed", 1


def main() -> None:
    environment = AdapterEnvironment.load()
    emitter = EventEmitter(environment, "adapter:photogrammetry:odx-v1")
    emitter.ready()
    stage = "startup"
    try:
        if Path("/code/VERSION").read_text().strip() != "3.8.3":
            raise ValueError("ODX installation must be version 3.8.3")
        stage = "input"
        emitter.emit("stage", stage_id=stage, state="started", checkpoint_id=None)
        if environment.task_config:
            raise ValueError("ODX task_config must be empty; parameters belong to the input manifest")
        work = environment.job_root / "work" / "odx-adapter"
        extracted = work / "input"
        safe_extract_tar(environment.input_path, extracted)
        input_path = extracted / "input-manifest.json"
        images = extracted / "images"
        input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
        validate_photogrammetry_odx_input(input_manifest)
        _verify_images(images, input_manifest)
        emitter.emit(
            "stage",
            stage_id="input",
            state="completed",
            checkpoint_id=f"checkpoint:input:{photogrammetry_odx_input_digest(input_manifest)[-24:]}",
        )

        stage = "odx"
        datasets = work / "datasets"
        project = datasets / "project"
        project.mkdir(parents=True)
        shutil.copytree(images, project / "images")
        emitter.emit("stage", stage_id=stage, state="started", checkpoint_id=None)
        log_path = work / "odx.log"
        with log_path.open("wb") as log:
            completed = subprocess.run(
                build_odx_command(datasets, input_manifest),
                # PDAL requires a home directory even when the container UID has no passwd entry.
                env={**os.environ, "HOME": str(environment.job_root)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"ODX exited with code {completed.returncode}")
        stage = "output"
        manifest = build_odx_output_manifest(project, input_manifest)
        emitter.emit(
            "stage",
            stage_id="odx",
            state="completed",
            checkpoint_id=manifest["checkpoint"]["checkpointId"],
        )

        artifacts = environment.job_root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        manifest_path = artifacts / "odx-output-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        bundle_path = artifacts / "odx-output.tar"
        create_odx_bundle(project, input_path, manifest_path, bundle_path)
        for path, media_type, role in [
            (project / "odm_orthophoto/odm_orthophoto.tif", "image/tiff", "odx-orthophoto"),
            (bundle_path, "application/x-tar", "odx-bundle"),
            (manifest_path, "application/json", "odx-manifest"),
        ]:
            reference = event_artifact_ref(path, environment.job_root, media_type, role)
            emitter.emit("artifact-candidate", stage_id="artifact", artifact=reference)
        emitter.emit(
            "result",
            status="ok",
            manifest=event_artifact_ref(
                manifest_path,
                environment.job_root,
                "application/json",
                "odx-manifest",
            ),
            error=None,
        )
    except (OSError, RuntimeError, tarfile.TarError, ValueError) as error:
        code, exit_code = classify_adapter_error(error, stage)
        emitter.result_error(code, str(error), stage)
        raise SystemExit(exit_code) from error


if __name__ == "__main__":
    main()
