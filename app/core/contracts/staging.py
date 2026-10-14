"""Validation for pre-publication Mesh staging manifests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from core.contracts._validation import (
    require_enum,
    require_finite_number,
    require_identity,
    require_keys,
    require_media_type,
    require_object,
    require_relative_path,
    require_sha256,
    require_string,
)
from core.contracts.publication import validate_candidate_catalog

STAGING_RE = re.compile(r"^staging:sha256:([0-9a-f]{64})$")


def validate_staging_manifest(manifest: Mapping[str, Any], catalog: Mapping[str, Any]) -> None:
    validate_candidate_catalog(catalog)
    root = require_object(manifest, "staging")
    require_keys(
        root,
        "staging",
        required={
            "schemaVersion",
            "stagingId",
            "candidateId",
            "source",
            "coordinates",
            "pipeline",
            "tileset",
            "files",
            "validation",
        },
    )
    if root["schemaVersion"] != "mesh-staging-manifest.v2":
        raise ValueError("schemaVersion must be mesh-staging-manifest.v2")
    match = STAGING_RE.fullmatch(root["stagingId"])
    if match is None:
        raise ValueError("stagingId must contain the output tree digest")
    candidate_id = require_identity(root["candidateId"], "candidateId")
    candidate = next((item for item in catalog["candidates"] if item["id"] == candidate_id), None)
    if candidate is None or candidate["role"] != "production-mesh":
        raise ValueError("candidateId must identify a production Mesh candidate")

    source = require_object(root["source"], "source")
    require_keys(source, "source", required={"odxOutputManifestDigest", "checkpointId"})
    require_sha256(source["odxOutputManifestDigest"], "source.odxOutputManifestDigest")
    require_identity(source["checkpointId"], "source.checkpointId")

    coordinates = require_object(root["coordinates"], "coordinates")
    require_keys(
        coordinates,
        "coordinates",
        required={"horizontal", "vertical", "axisOrder", "originOffsetMeters"},
    )
    horizontal = require_string(coordinates["horizontal"], "coordinates.horizontal")
    if re.fullmatch(r"EPSG:[1-9][0-9]*", horizontal) is None:
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

    pipeline = require_object(root["pipeline"], "pipeline")
    require_keys(
        pipeline,
        "pipeline",
        required={
            "modelVersion",
            "taskRevision",
            "basisUniversal",
            "texturePolicyDigest",
            "validationReport",
        },
    )
    if pipeline["modelVersion"] != "v55":
        raise ValueError("pipeline.modelVersion must be v55")
    if pipeline["taskRevision"] != "mesh.remesh-3d-tiles.v2":
        raise ValueError("pipeline.taskRevision must identify the v55 Worker")
    basisu = require_object(pipeline["basisUniversal"], "pipeline.basisUniversal")
    require_keys(
        basisu,
        "pipeline.basisUniversal",
        required={"version", "gitCommit", "sourceSha256", "binarySha256"},
    )
    if basisu["version"] != "1.16.3":
        raise ValueError("pipeline.basisUniversal.version must be 1.16.3")
    if basisu["gitCommit"] != "1531cfaf9ed5232248a0a45736686a849ca3befc":
        raise ValueError("pipeline.basisUniversal.gitCommit is not the locked source")
    if basisu["sourceSha256"] != ("b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac"):
        raise ValueError("pipeline.basisUniversal.sourceSha256 is not the locked source")
    if (
        not isinstance(basisu["binarySha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", basisu["binarySha256"]) is None
    ):
        raise ValueError("pipeline.basisUniversal.binarySha256 must be a bare SHA-256")
    require_sha256(pipeline["texturePolicyDigest"], "pipeline.texturePolicyDigest")
    validation_report = require_object(pipeline["validationReport"], "pipeline.validationReport")
    require_keys(
        validation_report,
        "pipeline.validationReport",
        required={"relativePath", "contentDigest", "payloadDigest"},
    )
    if validation_report["relativePath"] != "build-report.json":
        raise ValueError("pipeline.validationReport.relativePath must be build-report.json")
    require_sha256(validation_report["contentDigest"], "pipeline.validationReport.contentDigest")
    require_sha256(validation_report["payloadDigest"], "pipeline.validationReport.payloadDigest")

    tileset = require_object(root["tileset"], "tileset")
    require_keys(
        tileset,
        "tileset",
        required={"artifactId", "relativePath", "contentDigest", "contentTree"},
    )
    require_identity(tileset["artifactId"], "tileset.artifactId")
    if tileset["relativePath"] != "tileset.json":
        raise ValueError("tileset.relativePath must be tileset.json")
    require_sha256(tileset["contentDigest"], "tileset.contentDigest")
    tree = require_object(tileset["contentTree"], "tileset.contentTree")
    require_keys(tree, "tileset.contentTree", required={"algorithm", "contentDigest", "files", "bytes"})
    if tree["algorithm"] != "sha256-tree-v1":
        raise ValueError("tileset.contentTree.algorithm must be sha256-tree-v1")
    tree_digest = require_sha256(tree["contentDigest"], "tileset.contentTree.contentDigest")
    if root["stagingId"] != f"staging:{tree_digest}":
        raise ValueError("stagingId must match tileset.contentTree.contentDigest")
    _json_integer(tree["files"], "tileset.contentTree.files", minimum=1)
    tree_bytes = _json_integer(tree["bytes"], "tileset.contentTree.bytes", minimum=1)

    validation = require_object(root["validation"], "validation")
    require_keys(
        validation,
        "validation",
        required={"referencedContents", "missingContents", "outputBytes", "maxOutputBytes"},
    )
    _json_integer(validation["referencedContents"], "validation.referencedContents", minimum=1)
    missing = _json_integer(validation["missingContents"], "validation.missingContents", minimum=0)
    if missing != 0:
        raise ValueError("validation.missingContents must be zero")
    output_bytes = _json_integer(validation["outputBytes"], "validation.outputBytes", minimum=1)
    if output_bytes != tree_bytes:
        raise ValueError("validation.outputBytes must match the content tree")
    maximum = validation["maxOutputBytes"]
    if maximum is not None:
        maximum = _json_integer(maximum, "validation.maxOutputBytes", minimum=1)
    if maximum is not None and tree_bytes > maximum:
        raise ValueError("staged output exceeds maxOutputBytes")

    files = root["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("files must contain the complete staging inventory")
    seen: set[str] = set()
    total = 0
    for index, value in enumerate(files):
        path = f"files[{index}]"
        item = require_object(value, path)
        require_keys(item, path, required={"path", "size_bytes", "sha256", "content_type"})
        relative = require_relative_path(item["path"], f"{path}.path")
        if relative in seen:
            raise ValueError("files paths must be unique")
        seen.add(relative)
        total += _json_integer(item["size_bytes"], f"{path}.size_bytes", minimum=0)
        digest = item["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{path}.sha256 must be a bare SHA-256 digest")
        require_media_type(item["content_type"], f"{path}.content_type")
    if (
        "tileset.json" not in seen
        or validation_report["relativePath"] not in seen
        or len(files) != tree["files"]
        or total != tree_bytes
    ):
        raise ValueError("files inventory does not match the staging content tree")
    report_file = next(item for item in files if item["path"] == validation_report["relativePath"])
    if validation_report["contentDigest"] != f"sha256:{report_file['sha256']}":
        raise ValueError("pipeline.validationReport.contentDigest does not match files inventory")


def _json_integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, int):
        converted = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        converted = int(value)
    else:
        raise ValueError(f"{path} must be an integer")
    if converted < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return converted
