"""Atomic content-addressed promotion for validated Mesh staging output."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.artifacts import tree_identity
from core.contracts._validation import canonical_json_digest
from core.contracts.publication import validate_publication_manifest


def _require_owner(value: str, name: str) -> str:
    if (
        not value
        or len(value) > 128
        or not all(character.isalnum() or character in "._:-" for character in value)
    ):
        raise ValueError(f"{name} is not a stable identity")
    return value


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            directories.append(path)
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in reversed(directories):
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publication_manifest(
    staging: Mapping[str, Any],
    validation: Mapping[str, Any],
    identity: Mapping[str, Any],
    remesh_strategy_revision: str,
) -> dict[str, Any]:
    digest = identity["contentDigest"]
    digest_hex = digest.removeprefix("sha256:")
    publication_id = f"publication:{digest}"
    tileset = staging["tileset"]
    manifest = {
        "schemaVersion": "mesh-publication-manifest.v1",
        "publicationId": publication_id,
        "candidateId": staging["candidateId"],
        "state": "published",
        "source": {
            "odxOutputManifestDigest": staging["source"]["odxOutputManifestDigest"],
            "remeshStrategyRevision": remesh_strategy_revision,
        },
        "atomicDirectory": {
            "algorithm": identity["algorithm"],
            "treeDigest": digest,
            "files": identity["files"],
            "bytes": identity["bytes"],
            "directoryName": digest_hex,
        },
        "tileset": {
            "artifactId": tileset["artifactId"],
            "relativePath": "tileset.json",
            "contentDigest": tileset["contentDigest"],
            "mediaType": "application/json",
        },
        "validation": {
            "contract": "mesh-publication-validation.v1",
            "reportDigest": canonical_json_digest(validation),
        },
        "result": {
            "schemaVersion": "mesh-publication-result.v1",
            "publicationId": publication_id,
            "candidateId": staging["candidateId"],
            "tileset": {
                "artifactId": tileset["artifactId"],
                "contentDigest": tileset["contentDigest"],
                "accessPath": f"/v1/mesh-publications/{publication_id}/tileset.json",
                "authorization": "bearer",
            },
        },
    }
    return manifest


def _record_publication(
    registry_path: Path,
    manifest: Mapping[str, Any],
    owner_job_id: str,
    owner_attempt_id: str,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(registry_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS publications (
                publication_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS publication_claims (
                publication_id TEXT NOT NULL,
                owner_job_id TEXT NOT NULL,
                owner_attempt_id TEXT NOT NULL,
                PRIMARY KEY(publication_id, owner_job_id, owner_attempt_id),
                FOREIGN KEY(publication_id) REFERENCES publications(publication_id)
            );
            """
        )
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT manifest_json FROM publications WHERE publication_id=?",
            (manifest["publicationId"],),
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO publications(publication_id, manifest_json) VALUES(?, ?)",
                (manifest["publicationId"], encoded),
            )
        elif existing[0] != encoded:
            raise ValueError("publication identity already has a different manifest")
        connection.execute(
            "INSERT OR IGNORE INTO publication_claims(publication_id, owner_job_id, owner_attempt_id) "
            "VALUES(?, ?, ?)",
            (manifest["publicationId"], owner_job_id, owner_attempt_id),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def publish_staging(
    staging_root: Path,
    staging: Mapping[str, Any],
    validation: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    publications_root: Path,
    registry_path: Path,
    remesh_strategy_revision: str,
    owner_job_id: str,
    owner_attempt_id: str,
) -> dict[str, Any]:
    """Copy, verify, reveal, and durably register one immutable publication."""
    if (
        validation.get("schemaVersion") != "mesh-staging-validation.v1"
        or validation.get("passed") is not True
        or validation.get("candidateId") != staging.get("candidateId")
        or validation.get("stagingId") != staging.get("stagingId")
        or validation.get("contentTree") != staging.get("tileset", {}).get("contentTree")
    ):
        raise ValueError("staging validation report does not match the staging manifest")
    if len(remesh_strategy_revision) != 40 or any(
        character not in "0123456789abcdef" for character in remesh_strategy_revision
    ):
        raise ValueError("remesh_strategy_revision must be a full lowercase Git revision")
    owner_job_id = _require_owner(owner_job_id, "owner_job_id")
    owner_attempt_id = _require_owner(owner_attempt_id, "owner_attempt_id")
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise ValueError("staging root is missing or symbolic")

    expected = validation["contentTree"]
    publications_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=publications_root))
    final = publications_root / expected["contentDigest"].removeprefix("sha256:")
    try:
        shutil.copytree(staging_root, temporary, dirs_exist_ok=True, symlinks=True)
        observed = tree_identity(temporary)
        if observed != expected:
            raise ValueError("copied publication content identity does not match validation")
        manifest = _publication_manifest(staging, validation, observed, remesh_strategy_revision)
        validate_publication_manifest(manifest, catalog)
        _fsync_tree(temporary)
        if final.is_symlink():
            raise ValueError("publication path is symbolic link")
        if final.exists():
            if not final.is_dir():
                raise ValueError("publication path is not a directory")
            if tree_identity(final) != observed:
                raise ValueError("publication directory identity collision")
        else:
            try:
                os.rename(temporary, final)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                if final.is_symlink():
                    raise ValueError("publication path is symbolic link") from error
                if tree_identity(final) != observed:
                    raise ValueError("publication directory identity collision") from error
            directory = os.open(publications_root, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        _record_publication(registry_path, manifest, owner_job_id, owner_attempt_id)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--staging-manifest", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--publications-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--remesh-strategy-revision", required=True)
    parser.add_argument("--owner-job-id", required=True)
    parser.add_argument("--owner-attempt-id", required=True)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    manifest = publish_staging(
        args.staging_root,
        json.loads(args.staging_manifest.read_text()),
        json.loads(args.validation_report.read_text()),
        json.loads(args.catalog.read_text()),
        publications_root=args.publications_root,
        registry_path=args.registry,
        remesh_strategy_revision=args.remesh_strategy_revision,
        owner_job_id=args.owner_job_id,
        owner_attempt_id=args.owner_attempt_id,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
