from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from core.contracts.publication import validate_candidate_catalog, validate_publication_manifest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/mesh-candidates.json"
PUBLICATION = ROOT / "tests/fixtures/contracts/mesh-publication-manifest-v1.fixture.json"
SCHEMA = ROOT / "app/core/contracts/schemas/mesh-publication-manifest-v1.schema.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_catalog_namespaces_production_comparison_and_source_identities() -> None:
    catalog = load(CATALOG)
    validate_candidate_catalog(catalog)

    assert catalog["defaultProductionId"] == "mesh:remesh:v55-runtime"
    assert [(entry["id"], entry["role"]) for entry in catalog["candidates"]] == [
        ("comparison:terra:5.1.1", "comparison-baseline"),
        ("source:odx:3.8.3-eb9fd56", "photogrammetry-source"),
        ("mesh:remesh:v55-runtime", "production-mesh"),
    ]
    assert [entry["id"] for entry in catalog["candidates"] if entry["productionEligible"]] == [
        "mesh:remesh:v55-runtime",
    ]


def test_publication_binds_atomic_directory_and_authorized_tileset_result() -> None:
    catalog = load(CATALOG)
    publication = load(PUBLICATION)

    validate_publication_manifest(publication, catalog)
    assert list(Draft202012Validator(load(SCHEMA)).iter_errors(publication)) == []
    assert publication["candidateId"] == "mesh:remesh:v55-runtime"
    assert publication["tileset"]["relativePath"] == "tileset.json"
    assert publication["result"]["tileset"]["authorization"] == "bearer"
    assert publication["result"]["tileset"]["accessPath"].endswith("/tileset.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"candidateId": "remesh-v38"}), "candidateId"),
        (lambda value: value.update({"candidateId": "comparison:terra:5.1.1"}), "production"),
        (lambda value: value.update({"publicationId": "publication:sha256:" + "0" * 64}), "publicationId"),
        (lambda value: value["atomicDirectory"].update({"directoryName": "latest"}), "directoryName"),
        (lambda value: value["tileset"].update({"relativePath": "nested/tileset.json"}), "relativePath"),
        (
            lambda value: value["result"]["tileset"].update(
                {"accessPath": "https://host.invalid/tileset.json"}
            ),
            "accessPath",
        ),
        (
            lambda value: value["result"].update({"candidateId": "mesh:remesh:v55-other"}),
            "result.candidateId",
        ),
    ],
)
def test_publication_rejects_aliases_partial_roots_and_ambient_urls(mutate, message: str) -> None:
    catalog = load(CATALOG)
    publication = copy.deepcopy(load(PUBLICATION))
    mutate(publication)

    with pytest.raises(ValueError, match=message):
        validate_publication_manifest(publication, catalog)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda entry: entry.update({"id": "mesh:remesh:v38:quality"}), "identity"),
        (lambda entry: entry.update({"role": []}), "role"),
    ],
)
def test_catalog_rejects_ambiguous_candidate_identity_or_role(mutate, message: str) -> None:
    catalog = copy.deepcopy(load(CATALOG))
    mutate(catalog["candidates"][2])

    with pytest.raises(ValueError, match=message):
        validate_candidate_catalog(catalog)
