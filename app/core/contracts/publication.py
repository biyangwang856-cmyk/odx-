"""Validation for Mesh publication identity and access contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from core.contracts._validation import (
    require_identity,
    require_keys,
    require_object,
    require_positive_int,
    require_sha256,
    require_string,
)

CANDIDATE_FIELDS = {"id", "legacyId", "label", "role", "productionEligible"}
PUBLICATION_FIELDS = {
    "schemaVersion",
    "publicationId",
    "candidateId",
    "state",
    "source",
    "atomicDirectory",
    "tileset",
    "validation",
    "result",
}
PUBLICATION_RE = re.compile(r"^publication:sha256:([0-9a-f]{64})$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
NAMESPACED_CANDIDATE_RE = re.compile(r"^(?:comparison|source|mesh):[a-z0-9._-]+:[a-z0-9._-]+$")
ROLE_PREFIXES = {
    "comparison-baseline": "comparison:",
    "photogrammetry-source": "source:",
    "production-mesh": "mesh:",
}


def validate_candidate_catalog(catalog: Mapping[str, Any]) -> None:
    root = require_object(catalog, "catalog")
    require_keys(root, "catalog", required={"schemaVersion", "defaultProductionId", "candidates"})
    if root["schemaVersion"] != "mesh-candidate-identities.v1":
        raise ValueError("catalog schemaVersion must be mesh-candidate-identities.v1")
    default_id = require_identity(root["defaultProductionId"], "defaultProductionId")
    candidates = root["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("catalog.candidates must be a non-empty array")

    ids: set[str] = set()
    legacy_ids: set[str] = set()
    default_entry: Mapping[str, Any] | None = None
    for index, value in enumerate(candidates):
        path = f"candidates[{index}]"
        entry = require_object(value, path)
        require_keys(entry, path, required=CANDIDATE_FIELDS)
        candidate_id = require_identity(entry["id"], f"{path}.id")
        legacy_id = require_identity(entry["legacyId"], f"{path}.legacyId")
        require_string(entry["label"], f"{path}.label")
        role = require_string(entry["role"], f"{path}.role")
        if NAMESPACED_CANDIDATE_RE.fullmatch(candidate_id) is None:
            raise ValueError(f"{path}.id is not a valid namespaced candidate identity")
        if role not in ROLE_PREFIXES or not candidate_id.startswith(ROLE_PREFIXES[role]):
            raise ValueError(f"{path}.id namespace does not match role")
        production = entry["productionEligible"]
        if not isinstance(production, bool) or production != (role == "production-mesh"):
            raise ValueError(f"{path}.productionEligible does not match role")
        if candidate_id in ids or legacy_id in legacy_ids or candidate_id in legacy_ids or legacy_id in ids:
            raise ValueError(f"{path} candidate identities must be unique")
        ids.add(candidate_id)
        legacy_ids.add(legacy_id)
        if candidate_id == default_id:
            default_entry = entry
    if default_entry is None or default_entry["role"] != "production-mesh":
        raise ValueError("defaultProductionId must identify a production Mesh candidate")


def validate_publication_manifest(manifest: Mapping[str, Any], catalog: Mapping[str, Any]) -> None:
    validate_candidate_catalog(catalog)
    root = require_object(manifest, "publication")
    require_keys(root, "publication", required=PUBLICATION_FIELDS)
    if root["schemaVersion"] != "mesh-publication-manifest.v1":
        raise ValueError("schemaVersion must be mesh-publication-manifest.v1")
    publication_id = require_string(root["publicationId"], "publicationId")
    publication_match = PUBLICATION_RE.fullmatch(publication_id)
    if publication_match is None:
        raise ValueError("publicationId must contain the atomic directory SHA-256")
    candidate_id = require_identity(root["candidateId"], "candidateId")
    candidate = next(
        (entry for entry in catalog["candidates"] if entry["id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError("candidateId must use a registered namespaced identity")
    if candidate["role"] != "production-mesh" or not candidate["productionEligible"]:
        raise ValueError("candidateId must identify a production Mesh candidate")
    if root["state"] != "published":
        raise ValueError("publication state must be published")

    _validate_source(root["source"])
    _validate_atomic_directory(
        root["atomicDirectory"],
        publication_id,
        publication_match.group(1),
    )
    tileset = _validate_tileset(root["tileset"])
    _validate_validation(root["validation"])
    _validate_result(root["result"], publication_id, candidate_id, tileset)


def _validate_source(value: object) -> None:
    source = require_object(value, "source")
    require_keys(
        source,
        "source",
        required={"odxOutputManifestDigest", "remeshStrategyRevision"},
    )
    require_sha256(source["odxOutputManifestDigest"], "source.odxOutputManifestDigest")
    revision = require_string(source["remeshStrategyRevision"], "source.remeshStrategyRevision")
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("source.remeshStrategyRevision must be a full Git revision")


def _validate_atomic_directory(value: object, publication_id: str, digest_hex: str) -> None:
    directory = require_object(value, "atomicDirectory")
    require_keys(
        directory,
        "atomicDirectory",
        required={"algorithm", "treeDigest", "files", "bytes", "directoryName"},
    )
    if directory["algorithm"] != "sha256-tree-v1":
        raise ValueError("atomicDirectory.algorithm must be sha256-tree-v1")
    tree_digest = require_sha256(directory["treeDigest"], "atomicDirectory.treeDigest")
    if publication_id != f"publication:{tree_digest}":
        raise ValueError("publicationId must match atomicDirectory.treeDigest")
    if directory["directoryName"] != digest_hex:
        raise ValueError("atomicDirectory.directoryName must be the tree digest hex")
    require_positive_int(directory["files"], "atomicDirectory.files")
    require_positive_int(directory["bytes"], "atomicDirectory.bytes")


def _validate_tileset(value: object) -> Mapping[str, Any]:
    tileset = require_object(value, "tileset")
    require_keys(
        tileset,
        "tileset",
        required={"artifactId", "relativePath", "contentDigest", "mediaType"},
    )
    require_identity(tileset["artifactId"], "tileset.artifactId")
    if tileset["relativePath"] != "tileset.json":
        raise ValueError("tileset.relativePath must be tileset.json")
    require_sha256(tileset["contentDigest"], "tileset.contentDigest")
    if tileset["mediaType"] != "application/json":
        raise ValueError("tileset.mediaType must be application/json")
    return tileset


def _validate_validation(value: object) -> None:
    validation = require_object(value, "validation")
    require_keys(validation, "validation", required={"contract", "reportDigest"})
    if validation["contract"] != "mesh-publication-validation.v1":
        raise ValueError("validation.contract must be mesh-publication-validation.v1")
    require_sha256(validation["reportDigest"], "validation.reportDigest")


def _validate_result(
    value: object,
    publication_id: str,
    candidate_id: str,
    tileset: Mapping[str, Any],
) -> None:
    result = require_object(value, "result")
    require_keys(
        result,
        "result",
        required={"schemaVersion", "publicationId", "candidateId", "tileset"},
    )
    if result["schemaVersion"] != "mesh-publication-result.v1":
        raise ValueError("result.schemaVersion must be mesh-publication-result.v1")
    if result["publicationId"] != publication_id:
        raise ValueError("result.publicationId must match publicationId")
    if result["candidateId"] != candidate_id:
        raise ValueError("result.candidateId must match candidateId")
    access = require_object(result["tileset"], "result.tileset")
    require_keys(
        access,
        "result.tileset",
        required={"artifactId", "contentDigest", "accessPath", "authorization"},
    )
    if access["artifactId"] != tileset["artifactId"]:
        raise ValueError("result.tileset.artifactId must match tileset")
    if access["contentDigest"] != tileset["contentDigest"]:
        raise ValueError("result.tileset.contentDigest must match tileset")
    expected_path = f"/v1/mesh-publications/{publication_id}/tileset.json"
    if access["accessPath"] != expected_path:
        raise ValueError("result.tileset.accessPath must be the authorized publication route")
    if access["authorization"] != "bearer":
        raise ValueError("result.tileset.authorization must be bearer")
