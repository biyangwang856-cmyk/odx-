"""Validation for the odx-output-manifest.v2 contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from core.contracts._validation import (
    canonical_json_digest,
    require_enum,
    require_finite_number,
    require_identity,
    require_keys,
    require_media_type,
    require_object,
    require_positive_int,
    require_relative_path,
    require_sha256,
    require_string,
)
from core.contracts.photogrammetry_odx import (
    photogrammetry_odx_input_digest,
    validate_photogrammetry_odx_input,
)

ROOT_FIELDS = {
    "schemaVersion",
    "outputId",
    "input",
    "producer",
    "parametersDigest",
    "checkpoint",
    "coordinates",
    "artifacts",
}
FILE_FIELDS = {"artifactId", "relativePath", "contentDigest", "bytes", "mediaType"}
EPSG_RE = re.compile(r"^EPSG:[1-9][0-9]*$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_odx_output_manifest(manifest: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> None:
    """Validate the complete ODX result and its exact input lineage."""
    validate_photogrammetry_odx_input(input_manifest)
    root = require_object(manifest, "manifest")
    require_keys(root, "manifest", required=ROOT_FIELDS)
    if root["schemaVersion"] != "odx-output-manifest.v2":
        raise ValueError("schemaVersion must be odx-output-manifest.v2")
    require_identity(root["outputId"], "outputId")

    _validate_input(root["input"], input_manifest)
    _validate_producer(root["producer"])
    expected_parameters = canonical_json_digest(input_manifest["odxParameters"])
    if require_sha256(root["parametersDigest"], "parametersDigest") != expected_parameters:
        raise ValueError("parametersDigest does not match the ODX input parameters")
    _validate_checkpoint(root["checkpoint"])
    _validate_coordinates(root["coordinates"], input_manifest["crs"])
    _validate_artifacts(root["artifacts"])
    binding = odx_checkpoint_binding(root, input_manifest)
    if root["checkpoint"]["contentDigest"] != binding:
        raise ValueError("checkpoint.contentDigest does not match the ODX execution binding")
    suffix = binding[-24:]
    if root["checkpoint"]["checkpointId"] != f"checkpoint:odx:{suffix}":
        raise ValueError("checkpoint.checkpointId does not match checkpoint.contentDigest")
    if root["outputId"] != f"odx-output:{suffix}":
        raise ValueError("outputId does not match checkpoint.contentDigest")


def odx_checkpoint_binding(manifest: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> str:
    """Bind a reusable ODX checkpoint to its complete execution identity."""
    return canonical_json_digest(
        {
            "input": photogrammetry_odx_input_digest(input_manifest),
            "producer": manifest["producer"],
            "parameters": input_manifest["odxParameters"],
            "coordinates": manifest["coordinates"],
            "artifacts": manifest["artifacts"],
        }
    )


def remesh_sources(manifest: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> dict[str, object]:
    """Project the only ODX fields that a Remesh adapter may consume."""
    validate_odx_output_manifest(manifest, input_manifest)
    return {
        "inputManifestDigest": manifest["input"]["manifestDigest"],
        "mesh": deepcopy(manifest["artifacts"]["mesh"]),
        "coordinates": deepcopy(manifest["coordinates"]),
        "checkpointId": manifest["checkpoint"]["checkpointId"],
    }


def _validate_input(value: object, input_manifest: Mapping[str, Any]) -> None:
    source = require_object(value, "input")
    require_keys(
        source,
        "input",
        required={"manifestDigest", "inputId", "imageContentDigest"},
    )
    if require_sha256(source["manifestDigest"], "input.manifestDigest") != (
        photogrammetry_odx_input_digest(input_manifest)
    ):
        raise ValueError("input.manifestDigest does not identify the supplied input manifest")
    if require_identity(source["inputId"], "input.inputId") != input_manifest["inputId"]:
        raise ValueError("input.inputId does not match the supplied input manifest")
    if (
        require_sha256(source["imageContentDigest"], "input.imageContentDigest")
        != (input_manifest["images"]["contentDigest"])
    ):
        raise ValueError("input.imageContentDigest does not match the supplied image tree")


def _validate_producer(value: object) -> None:
    producer = require_object(value, "producer")
    require_keys(
        producer,
        "producer",
        required={"engine", "engineVersion", "imageDigest", "sourceRevision"},
    )
    if producer["engine"] != "odx":
        raise ValueError("producer.engine must be odx")
    version = require_string(producer["engineVersion"], "producer.engineVersion")
    if VERSION_RE.fullmatch(version) is None:
        raise ValueError("producer.engineVersion must be a semantic version")
    require_sha256(producer["imageDigest"], "producer.imageDigest")
    revision = require_string(producer["sourceRevision"], "producer.sourceRevision")
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("producer.sourceRevision must be a full Git revision")


def _validate_checkpoint(value: object) -> None:
    checkpoint = require_object(value, "checkpoint")
    require_keys(
        checkpoint,
        "checkpoint",
        required={"checkpointId", "stage", "contentDigest"},
    )
    require_identity(checkpoint["checkpointId"], "checkpoint.checkpointId")
    if checkpoint["stage"] != "odm_orthophoto":
        raise ValueError("checkpoint.stage must be odm_orthophoto")
    require_sha256(checkpoint["contentDigest"], "checkpoint.contentDigest")


def _validate_coordinates(value: object, input_crs: object) -> None:
    coordinates = require_object(value, "coordinates")
    require_keys(
        coordinates,
        "coordinates",
        required={"horizontal", "vertical", "axisOrder", "originOffsetMeters"},
    )
    horizontal = require_string(coordinates["horizontal"], "coordinates.horizontal")
    if EPSG_RE.fullmatch(horizontal) is None:
        raise ValueError("coordinates.horizontal must be an EPSG identity")
    require_enum(
        coordinates["vertical"],
        "coordinates.vertical",
        {"ellipsoidal", "orthometric", "unspecified"},
    )
    require_enum(
        coordinates["axisOrder"],
        "coordinates.axisOrder",
        {"east,north,up", "longitude,latitude,height"},
    )
    offset = coordinates["originOffsetMeters"]
    if not isinstance(offset, list) or len(offset) != 3:
        raise ValueError("coordinates.originOffsetMeters must contain three numbers")
    for index, component in enumerate(offset):
        require_finite_number(component, f"coordinates.originOffsetMeters[{index}]")
    expected = require_object(input_crs, "input.crs")
    for name in ("horizontal", "vertical", "axisOrder"):
        if coordinates[name] != expected[name]:
            raise ValueError(f"coordinates.{name} does not match the ODX input CRS")


def _validate_artifacts(value: object) -> None:
    artifacts = require_object(value, "artifacts")
    require_keys(
        artifacts,
        "artifacts",
        required={"mesh", "orthophoto"},
    )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    mesh = require_object(artifacts["mesh"], "artifacts.mesh")
    require_keys(mesh, "artifacts.mesh", required={"obj", "mtl", "textures"})
    _validate_file(mesh["obj"], "artifacts.mesh.obj", seen_ids, seen_paths)
    _validate_file(mesh["mtl"], "artifacts.mesh.mtl", seen_ids, seen_paths)
    _validate_file_list(mesh["textures"], "artifacts.mesh.textures", seen_ids, seen_paths)
    _validate_file(artifacts["orthophoto"], "artifacts.orthophoto", seen_ids, seen_paths)
    if artifacts["orthophoto"]["mediaType"] != "image/tiff":
        raise ValueError("artifacts.orthophoto.mediaType must be image/tiff")


def _validate_file_list(value: object, path: str, seen_ids: set[str], seen_paths: set[str]) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain at least one artifact")
    for index, artifact in enumerate(value):
        _validate_file(artifact, f"{path}[{index}]", seen_ids, seen_paths)


def _validate_file(value: object, path: str, seen_ids: set[str], seen_paths: set[str]) -> None:
    artifact = require_object(value, path)
    require_keys(artifact, path, required=FILE_FIELDS)
    artifact_id = require_identity(artifact["artifactId"], f"{path}.artifactId")
    relative_path = require_relative_path(artifact["relativePath"], f"{path}.relativePath")
    require_sha256(artifact["contentDigest"], f"{path}.contentDigest")
    require_positive_int(artifact["bytes"], f"{path}.bytes")
    require_media_type(artifact["mediaType"], f"{path}.mediaType")
    if artifact_id in seen_ids:
        raise ValueError(f"{path}.artifactId must be unique")
    if relative_path in seen_paths:
        raise ValueError(f"{path}.relativePath must be unique")
    seen_ids.add(artifact_id)
    seen_paths.add(relative_path)
