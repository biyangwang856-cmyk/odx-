"""One-shot adapter from an ODX bundle to staged Mesh 3D Tiles."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.artifacts import artifact_ref, safe_extract_tar, sha256_file, tree_identity
from core.contracts._validation import (
    require_identity,
    require_positive_int,
    require_relative_path,
    require_sha256,
)
from core.contracts.odx_output import remesh_sources, validate_odx_output_manifest
from core.contracts.photogrammetry_odx import validate_photogrammetry_odx_input
from core.contracts.publication import validate_candidate_catalog
from core.contracts.staging import validate_staging_manifest
from core.pipelines.remesh import RemeshProfile, build_publication
from core.validation.odx_output import verify_output_files

from model_engine.common import AdapterEnvironment, EventEmitter, event_artifact_ref

CONFIG_FIELDS = {"odxManifestDigest", "candidateId", "featuresPerTile", "maxOutputBytes"}
BROWSER_TEXTURE_MAX_SIZE = 8192


def parse_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Remesh task_config must be an object")
    if set(value) != CONFIG_FIELDS:
        raise ValueError("Remesh task_config fields do not match the allowlist")
    parsed = dict(value)
    require_sha256(parsed["odxManifestDigest"], "odxManifestDigest")
    candidate = require_identity(parsed["candidateId"], "candidateId")
    if not candidate.startswith("mesh:"):
        raise ValueError("candidateId must be a namespaced Mesh candidate")
    require_positive_int(parsed["featuresPerTile"], "featuresPerTile", maximum=4096)
    if parsed["maxOutputBytes"] is not None:
        require_positive_int(parsed["maxOutputBytes"], "maxOutputBytes")
    return parsed


def build_remesh_command(
    input_obj: Path,
    output_dir: Path,
    odx_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    config = parse_config(config)
    coordinates = odx_manifest["coordinates"]
    proj_data = Path(sys.prefix) / "share/proj"
    if not (proj_data / "proj.db").is_file():
        raise FileNotFoundError(proj_data / "proj.db")
    return [
        sys.executable,
        "-m",
        "core.pipelines.source_leaf_tiler",
        "--input-obj",
        str(input_obj.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--crs",
        coordinates["horizontal"],
        "--offset",
        *(str(value) for value in coordinates["originOffsetMeters"]),
        "--proj-data",
        str(proj_data),
        "--features-per-tile",
        str(config["featuresPerTile"]),
        "--crop-textures-dir",
        str((output_dir.parent / "texture-crops").resolve()),
        "--max-texture-size",
        str(BROWSER_TEXTURE_MAX_SIZE),
    ]


def resolve_executable(environment_name: str, default: str, label: str) -> str:
    executable = os.environ.get(environment_name) or shutil.which(default)
    if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"{label} executable is not configured")
    return str(Path(executable).resolve())


def resolve_gltf_transform() -> str:
    return resolve_executable("GLTF_TRANSFORM", "gltf-transform", "gltf-transform")


def resolve_basisu() -> str:
    return resolve_executable("BASISU", "basisu", "Basis Universal 1.16.3")


def resolve_node() -> str:
    return resolve_executable("NODE", "node", "Node.js")


def resolve_sharp_module() -> str:
    configured = os.environ.get("SHARP_MODULE", "/opt/gltf-transform/node_modules/sharp")
    module = Path(configured)
    if not module.is_dir() or not (module / "package.json").is_file():
        raise FileNotFoundError("locked Sharp module is not configured")
    return str(module.resolve())


def resolve_basisu_provenance() -> str:
    configured = Path(os.environ.get("BASISU_PROVENANCE", "/opt/basisu/provenance.json"))
    if not configured.is_file():
        raise FileNotFoundError("Basis Universal source provenance is not configured")
    return str(configured.resolve())


def read_v55_pipeline_provenance(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "build-report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload_digest = report.get("reportPayloadDigest")
    payload = dict(report)
    payload.pop("reportPayloadDigest", None)
    observed_payload_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    if payload_digest != observed_payload_digest:
        raise ValueError("v55 build report payload digest is invalid")
    gpu = report.get("gpuTextures")
    if not isinstance(gpu, dict) or gpu.get("codec") != "etc1s":
        raise ValueError("v55 build report has no ETC1S texture validation")
    if gpu.get("geometryAndUvBytesPreserved") is not True:
        raise ValueError("v55 build report did not preserve geometry and UV bytes")
    models = gpu.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("v55 build report has no model validation rows")
    toolchains = {json.dumps(row.get("toolchain"), sort_keys=True) for row in models}
    if len(toolchains) != 1 or "null" in toolchains:
        raise ValueError("v55 model reports do not share one locked toolchain")
    toolchain = json.loads(next(iter(toolchains)))
    required = {
        "basisUniversalVersion",
        "basisUniversalGitCommit",
        "basisUniversalSourceSha256",
        "basisUniversalBinarySha256",
        "policySha256",
    }
    if set(toolchain) != required:
        raise ValueError("v55 toolchain provenance fields are invalid")
    return {
        "modelVersion": "v55",
        "taskRevision": "mesh.remesh-3d-tiles.v2",
        "basisUniversal": {
            "version": toolchain["basisUniversalVersion"],
            "gitCommit": toolchain["basisUniversalGitCommit"],
            "sourceSha256": toolchain["basisUniversalSourceSha256"],
            "binarySha256": toolchain["basisUniversalBinarySha256"],
        },
        "texturePolicyDigest": f"sha256:{toolchain['policySha256']}",
        "validationReport": {
            "relativePath": "build-report.json",
            "contentDigest": sha256_file(report_path),
            "payloadDigest": payload_digest,
        },
    }


def build_staging_manifest(
    output_dir: Path,
    odx_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    config = parse_config(config)
    validate_candidate_catalog(catalog)
    candidate = next(
        (item for item in catalog["candidates"] if item["id"] == config["candidateId"]),
        None,
    )
    if candidate is None or candidate["role"] != "production-mesh":
        raise ValueError("candidateId must identify a production Mesh candidate")
    tileset_path = output_dir / "tileset.json"
    tileset = json.loads(tileset_path.read_text(encoding="utf-8"))
    if not isinstance(tileset, dict) or not isinstance(tileset.get("root"), dict):
        raise ValueError("staged tileset.json has no root tile")
    references: list[str] = []

    def visit(tile: dict[str, Any]) -> None:
        contents: list[object] = []
        if "content" in tile and "contents" in tile:
            raise ValueError("tileset tile cannot contain both content and contents")
        if "content" in tile:
            contents.append(tile["content"])
        if "contents" in tile:
            multiple = tile["contents"]
            if not isinstance(multiple, list):
                raise ValueError("tileset contents must be an array")
            contents.extend(multiple)
        for content in contents:
            if not isinstance(content, dict):
                raise ValueError("tileset content must be an object")
            uri = content.get("uri", content.get("url"))
            if not isinstance(uri, str) or not uri:
                raise ValueError("tileset content URI is invalid")
            relative = Path(uri)
            if relative.is_absolute() or ".." in relative.parts or "\\" in uri:
                raise ValueError("tileset content URI escaped staging")
            path = output_dir / relative
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            references.append(uri)
        children = tile.get("children", [])
        if not isinstance(children, list):
            raise ValueError("tileset children must be an array")
        for child in children:
            if not isinstance(child, dict):
                raise ValueError("tileset child must be an object")
            visit(child)

    visit(tileset["root"])
    if not references:
        raise ValueError("staged tileset has no renderable content")
    pipeline = read_v55_pipeline_provenance(output_dir)
    tree = tree_identity(output_dir)
    if config["maxOutputBytes"] is not None and tree["bytes"] > config["maxOutputBytes"]:
        raise ValueError("staged output exceeds maxOutputBytes")
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        content_type = (
            "model/gltf-binary"
            if path.suffix.lower() == ".glb"
            else mimetypes.guess_type(relative)[0] or "application/octet-stream"
        )
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
                "content_type": content_type,
            }
        )
    manifest = {
        "schemaVersion": "mesh-staging-manifest.v2",
        "stagingId": f"staging:{tree['contentDigest']}",
        "candidateId": config["candidateId"],
        "source": {
            "odxOutputManifestDigest": config["odxManifestDigest"],
            "checkpointId": odx_manifest["checkpoint"]["checkpointId"],
        },
        "coordinates": odx_manifest["coordinates"],
        "pipeline": pipeline,
        "tileset": {
            "artifactId": artifact_ref(
                tileset_path,
                output_dir,
                "application/json",
                "staging-tileset",
            )["artifactId"],
            "relativePath": "tileset.json",
            "contentDigest": sha256_file(tileset_path),
            "contentTree": tree,
        },
        "files": files,
        "validation": {
            "referencedContents": len(set(references)),
            "missingContents": 0,
            "outputBytes": tree["bytes"],
            "maxOutputBytes": config["maxOutputBytes"],
        },
    }
    validate_staging_manifest(manifest, catalog)
    return manifest


def create_staging_bundle(tiles: Path, manifest_path: Path, bundle_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = tiles.resolve()
    with tarfile.open(bundle_path, "w") as bundle:
        for item in manifest["files"]:
            relative = Path(require_relative_path(item["path"], "staging file path"))
            path = tiles / relative
            cursor = tiles
            has_symlinked_parent = False
            for part in relative.parts[:-1]:
                cursor /= part
                has_symlinked_parent |= cursor.is_symlink()
            if (
                has_symlinked_parent
                or path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(root)
                or sha256_file(path) != "sha256:" + item["sha256"]
            ):
                raise ValueError(f"staging file changed before archive: {item['path']}")
            bundle.add(path, arcname=item["path"], recursive=False)
        bundle.add(manifest_path, arcname="mesh-staging-manifest.json")


def main() -> None:
    environment = AdapterEnvironment.load()
    emitter = EventEmitter(environment, "adapter:mesh:remesh-v1")
    emitter.ready()
    stage = "input"
    try:
        config = parse_config(environment.task_config)
        catalog = json.loads((Path(__file__).parents[2] / "configs/mesh-candidates.json").read_text())
        work = environment.job_root / "work" / "remesh-adapter"
        extracted = work / "input"
        safe_extract_tar(environment.input_path, extracted)
        input_manifest = json.loads((extracted / "input-manifest.json").read_text())
        odx_manifest_path = extracted / "odx-output-manifest.json"
        odx_manifest = json.loads(odx_manifest_path.read_text())
        validate_photogrammetry_odx_input(input_manifest)
        if sha256_file(odx_manifest_path) != config["odxManifestDigest"]:
            raise ValueError("ODX manifest digest does not match task_config")
        validate_odx_output_manifest(odx_manifest, input_manifest)
        verify_output_files(extracted, odx_manifest)
        sources = remesh_sources(odx_manifest, input_manifest)
        input_obj = extracted / sources["mesh"]["obj"]["relativePath"]
        if input_obj.is_symlink() or not input_obj.is_file():
            raise FileNotFoundError(input_obj)
        emitter.emit(
            "stage",
            stage_id=stage,
            state="completed",
            checkpoint_id=sources["checkpointId"],
        )

        stage = "tiling"
        source_output = work / "tiles-source"
        output = work / "tiles"
        emitter.emit("stage", stage_id=stage, state="started", checkpoint_id=None)
        log_path = work / "remesh.log"
        with log_path.open("wb") as log:
            completed = subprocess.run(
                build_remesh_command(input_obj, source_output, odx_manifest, config),
                cwd=Path(__file__).parents[2],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                start_new_session=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Remesh tiler exited with code {completed.returncode}")
        build_publication(
            profile=RemeshProfile(
                **json.loads((Path(__file__).parents[2] / "configs/server/remesh-profile.json").read_text())
            ),
            source=source_output,
            output=output,
            executable=resolve_gltf_transform(),
            workers=4,
            node_executable=resolve_node(),
            basisu_executable=resolve_basisu(),
            sharp_module=resolve_sharp_module(),
            basisu_provenance=resolve_basisu_provenance(),
        )
        staging = build_staging_manifest(output, odx_manifest, config, catalog)
        emitter.emit(
            "stage",
            stage_id=stage,
            state="completed",
            checkpoint_id=staging["stagingId"].replace("staging:", "checkpoint:staging:"),
        )

        artifacts = environment.job_root / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        manifest_path = artifacts / "mesh-staging-manifest.json"
        manifest_path.write_text(json.dumps(staging, indent=2) + "\n")
        bundle_path = artifacts / "mesh-staging.tar"
        create_staging_bundle(output, manifest_path, bundle_path)
        for path, media_type, role in [
            (bundle_path, "application/x-tar", "mesh-staging-bundle"),
            (manifest_path, "application/json", "mesh-staging-manifest"),
        ]:
            emitter.emit(
                "artifact-candidate",
                stage_id="artifact",
                artifact=event_artifact_ref(path, environment.job_root, media_type, role),
            )
        emitter.emit(
            "result",
            status="ok",
            manifest=event_artifact_ref(
                manifest_path,
                environment.job_root,
                "application/json",
                "mesh-staging-manifest",
            ),
            error=None,
        )
    except (OSError, RuntimeError, tarfile.TarError, ValueError) as error:
        code = "invalid_input" if isinstance(error, (FileNotFoundError, ValueError)) else "adapter_failed"
        emitter.result_error(code, str(error), stage)
        raise SystemExit(2 if code == "invalid_input" else 1) from error


if __name__ == "__main__":
    main()
