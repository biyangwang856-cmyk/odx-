#!/usr/bin/env python3
"""Inventory the frozen Phase 0 publications without treating gaps as passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from core.artifacts import image_tree_identity

from experiments.analysis.candidate_protocol import protocol_digest, validate_protocol

EVIDENCE_GAPS = ("structure", "browser-profile", "screenshots", "http-range-cors")
MAX_TILESET_BYTES = 16 * 1024 * 1024


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_root_tileset(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    asset = document.get("asset")
    geometric_error = document.get("geometricError")
    return (
        isinstance(asset, dict)
        and isinstance(asset.get("version"), str)
        and bool(asset["version"].strip())
        and _finite_number(geometric_error)
        and isinstance(document.get("root"), dict)
    )


def _publication_identity(root: Path) -> dict[str, object]:
    identity = image_tree_identity(root)
    return {
        "algorithm": identity["algorithm"],
        "files": identity["count"],
        "bytes": identity["bytes"],
        "digest": identity["digest"],
    }


def _inspect_root(
    configured_root: str | None,
    expected_publication: dict[str, object],
) -> tuple[str | None, str, str | None, dict[str, object] | None]:
    if not configured_root:
        return None, "missing-root-environment", None, None

    try:
        parsed = urlsplit(configured_root)
    except ValueError:
        return configured_root, "unreachable", None, None
    if parsed.scheme in {"http", "https"}:
        root = configured_root.rstrip("/")
        try:
            request = Request(f"{root}/tileset.json", headers={"Accept": "application/json"})
            with urlopen(request, timeout=10) as response:
                payload = response.read(MAX_TILESET_BYTES + 1)
        except HTTPError as error:
            state = "missing-tileset" if error.code == 404 else "unreachable"
            return root, state, None, None
        except (IncompleteRead, OSError, TimeoutError, URLError, ValueError):
            return root, "unreachable", None, None
        observed_publication = None
        ready_state = "identity-unverified"
    else:
        try:
            root_path = Path(configured_root).expanduser().resolve()
        except (OSError, RuntimeError):
            return configured_root, "unreachable", None, None
        root = str(root_path)
        try:
            if not root_path.is_dir():
                return root, "missing-root", None, None
            tileset = root_path / "tileset.json"
            if not tileset.is_file():
                return root, "missing-tileset", None, None
            if tileset.stat().st_size > MAX_TILESET_BYTES:
                return root, "invalid-tileset", None, None
            payload = tileset.read_bytes()
            observed_publication = _publication_identity(root_path)
        except OSError:
            return root, "unreachable", None, None
        except ValueError:
            return root, "invalid-publication", None, None
        ready_state = "ready" if observed_publication == expected_publication else "publication-mismatch"

    if len(payload) > MAX_TILESET_BYTES:
        return root, "invalid-tileset", None, observed_publication
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return root, "invalid-tileset", None, observed_publication
    if not _valid_root_tileset(document):
        return root, "invalid-tileset", None, observed_publication
    return (
        root,
        ready_state,
        f"sha256:{hashlib.sha256(payload).hexdigest()}",
        observed_publication,
    )


def inventory_candidates(protocol: dict[str, Any], environment: Mapping[str, str]) -> dict[str, object]:
    """Resolve candidate roots and preserve every unavailable evidence edge."""
    validate_protocol(protocol)
    rows: list[dict[str, object]] = []
    for candidate in protocol["candidates"]:
        root_environment = candidate["rootEnvironment"]
        configured_root = environment.get(root_environment)
        root, availability, tileset_digest, observed_publication = _inspect_root(
            configured_root,
            candidate["expectedPublication"],
        )

        gaps = list(EVIDENCE_GAPS)
        if availability != "ready":
            gaps.insert(
                0,
                "publication-identity"
                if availability in {"identity-unverified", "publication-mismatch"}
                else "publication-root",
            )
        rows.append(
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "kind": candidate["kind"],
                "productionEligible": candidate["productionEligible"],
                "revisionKind": candidate["revisionKind"],
                "revision": candidate["revision"],
                "inputSource": candidate["inputSource"],
                "buildSource": candidate.get("buildSource"),
                "expectedPublication": candidate["expectedPublication"],
                "rootEnvironment": root_environment,
                "root": root,
                "availability": availability,
                "tilesetDigest": tileset_digest,
                "observedPublication": observed_publication,
                "evidenceGaps": gaps,
            }
        )

    return {
        "schemaVersion": "mesh-candidate-inventory.v1",
        "protocolDigest": protocol_digest(protocol),
        "datasetId": protocol["dataset"]["id"],
        "sourceImages": protocol["dataset"]["images"],
        "comparisonBaselineId": protocol["roles"]["comparisonBaselineId"],
        "photogrammetrySourceId": protocol["roles"]["photogrammetrySourceId"],
        "productionMeshCandidateIds": protocol["roles"]["productionMeshCandidateIds"],
        "recommendedProductionMeshId": protocol["roles"]["recommendedProductionMeshId"],
        "candidates": rows,
        "allCandidatesVerified": all(row["availability"] == "ready" for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = inventory_candidates(protocol, os.environ)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
