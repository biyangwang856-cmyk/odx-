"""Validation for the photogrammetry.odx.v1 input contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from core.contracts._validation import (
    canonical_json_digest,
    require_enum,
    require_identity,
    require_keys,
    require_media_type,
    require_object,
    require_positive_int,
    require_sha256,
    require_string,
)

ROOT_REQUIRED = {
    "schemaVersion",
    "inputId",
    "acquisition",
    "images",
    "crs",
    "resourceBudget",
    "odxParameters",
}
OPTIONAL_INPUTS = {"groundControl", "geoFile", "imageGroups"}
ARTIFACT_FIELDS = {"artifactId", "contentDigest", "bytes", "mediaType"}
PARAMETER_RULES: dict[str, tuple[str, object]] = {
    "orthophotoMode": ("enum", {"3d", "2.5d"}),
    "featureQuality": ("enum", {"ultra", "high", "medium", "low", "lowest"}),
    "matcherNeighbors": ("integer", 64),
    "pcQuality": ("enum", {"ultra", "high", "medium", "low", "lowest"}),
    "meshSize": ("integer-range", (1_000, 10_000_000)),
    "meshOctreeDepth": ("integer-range", (8, 14)),
    "texturingToneMapping": ("enum", {"none", "gamma", "drago", "reinhard"}),
}
RFC3339_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
EPSG_RE = re.compile(r"^EPSG:[1-9][0-9]*$")


def validate_photogrammetry_odx_input(manifest: Mapping[str, Any]) -> None:
    """Validate one command-free ODX input manifest."""
    root = require_object(manifest, "manifest")
    require_keys(root, "manifest", required=ROOT_REQUIRED, optional=OPTIONAL_INPUTS)
    if root["schemaVersion"] != "photogrammetry.odx.v1":
        raise ValueError("schemaVersion must be photogrammetry.odx.v1")
    input_id = require_identity(root["inputId"], "inputId")

    acquisition = require_object(root["acquisition"], "acquisition")
    require_keys(
        acquisition,
        "acquisition",
        required={"id"},
        optional={"capturedAt", "sensorId"},
    )
    if require_identity(acquisition["id"], "acquisition.id") != input_id:
        raise ValueError("acquisition.id must equal inputId")
    if "sensorId" in acquisition:
        require_identity(acquisition["sensorId"], "acquisition.sensorId")
    if "capturedAt" in acquisition:
        captured_at = require_string(acquisition["capturedAt"], "acquisition.capturedAt")
        if RFC3339_RE.fullmatch(captured_at) is None:
            raise ValueError("acquisition.capturedAt must be an RFC 3339 timestamp")
        try:
            parsed = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("acquisition.capturedAt must be an RFC 3339 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("acquisition.capturedAt must include a timezone")

    image_bytes, image_count = _validate_images(root["images"])
    optional_bytes = sum(_validate_artifact(root[name], name) for name in OPTIONAL_INPUTS if name in root)
    _validate_crs(root["crs"])
    max_images, max_input_bytes = _validate_budget(root["resourceBudget"])
    if image_count > max_images:
        raise ValueError("images.fileCount exceeds resourceBudget.maxImages")
    if image_bytes + optional_bytes > max_input_bytes:
        raise ValueError("input bytes exceed resourceBudget.maxInputBytes")
    _validate_parameters(root["odxParameters"])


def photogrammetry_odx_input_digest(manifest: Mapping[str, Any]) -> str:
    """Return the identity consumed by downstream output manifests."""
    validate_photogrammetry_odx_input(manifest)
    return canonical_json_digest(manifest)


def _validate_artifact(value: object, path: str) -> int:
    artifact = require_object(value, path)
    require_keys(artifact, path, required=ARTIFACT_FIELDS)
    require_identity(artifact["artifactId"], f"{path}.artifactId")
    require_sha256(artifact["contentDigest"], f"{path}.contentDigest")
    size = require_positive_int(artifact["bytes"], f"{path}.bytes")
    require_media_type(artifact["mediaType"], f"{path}.mediaType")
    return size


def _validate_images(value: object) -> tuple[int, int]:
    images = require_object(value, "images")
    require_keys(
        images,
        "images",
        required=ARTIFACT_FIELDS | {"fileCount", "treeAlgorithm"},
    )
    image_bytes = _validate_artifact(
        {name: images[name] for name in ARTIFACT_FIELDS},
        "images",
    )
    image_count = require_positive_int(images["fileCount"], "images.fileCount")
    if images["treeAlgorithm"] != "sha256-tree-v1":
        raise ValueError("images.treeAlgorithm must be sha256-tree-v1")
    if images["mediaType"] != "application/vnd.surveying.image-tree":
        raise ValueError("images.mediaType must identify an image tree")
    return image_bytes, image_count


def _validate_crs(value: object) -> None:
    crs = require_object(value, "crs")
    require_keys(crs, "crs", required={"horizontal", "vertical", "axisOrder"})
    horizontal = require_string(crs["horizontal"], "crs.horizontal")
    if EPSG_RE.fullmatch(horizontal) is None:
        raise ValueError("crs.horizontal must be an EPSG:<positive integer> identity")
    require_enum(crs["vertical"], "crs.vertical", {"ellipsoidal", "orthometric", "unspecified"})
    require_enum(
        crs["axisOrder"],
        "crs.axisOrder",
        {"east,north,up", "longitude,latitude,height"},
    )


def _validate_budget(value: object) -> tuple[int, int]:
    budget = require_object(value, "resourceBudget")
    fields = {
        "maxImages",
        "maxInputBytes",
        "maxWallSeconds",
        "maxCpuCores",
        "maxMemoryBytes",
        "maxDiskBytes",
    }
    require_keys(budget, "resourceBudget", required=fields)
    values = {name: require_positive_int(budget[name], f"resourceBudget.{name}") for name in fields}
    return values["maxImages"], values["maxInputBytes"]


def _validate_parameters(value: object) -> None:
    parameters = require_object(value, "odxParameters")
    require_keys(parameters, "odxParameters", required=set(), optional=set(PARAMETER_RULES))
    for name, raw in parameters.items():
        kind, constraint = PARAMETER_RULES[name]
        if kind == "enum":
            require_enum(raw, f"odxParameters.{name}", constraint)
        elif kind == "integer":
            require_positive_int(raw, f"odxParameters.{name}", maximum=constraint)
        else:
            minimum, maximum = constraint
            observed = require_positive_int(raw, f"odxParameters.{name}", maximum=maximum)
            if observed < minimum:
                raise ValueError(f"odxParameters.{name} must be at least {minimum}")
