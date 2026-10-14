from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from core.contracts.odx_output import remesh_sources, validate_odx_output_manifest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
INPUT = ROOT / "tests/fixtures/contracts/photogrammetry-odx-v1.lglglg.json"
OUTPUT = ROOT / "tests/fixtures/contracts/odx-output-manifest-v2.fixture.json"
SCHEMA = ROOT / "app/core/contracts/schemas/odx-output-manifest-v2.schema.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_errors(manifest: dict[str, object]) -> list[object]:
    return list(Draft202012Validator(load(SCHEMA)).iter_errors(manifest))


def test_odx_output_binds_engine_input_parameters_checkpoint_and_artifacts() -> None:
    input_manifest = load(INPUT)
    output_manifest = load(OUTPUT)

    validate_odx_output_manifest(output_manifest, input_manifest)
    assert schema_errors(output_manifest) == []
    assert output_manifest["producer"] == {
        "engine": "odx",
        "engineVersion": "3.8.3",
        "imageDigest": "sha256:" + "e" * 64,
        "sourceRevision": "eb9fd56ac430fc3735401d88aaef9945ebd00e14",
    }
    assert output_manifest["checkpoint"]["checkpointId"] == "checkpoint:odx:9e1bbc5ad26ecb0c00c70361"
    assert output_manifest["artifacts"]["orthophoto"]["relativePath"] == "odm_orthophoto/odm_orthophoto.tif"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"command": ["odm", "--rerun-all"]}), "unknown fields"),
        (lambda value: value["producer"].update({"imageDigest": "latest"}), "imageDigest"),
        (
            lambda value: value["input"].update({"manifestDigest": "sha256:" + "0" * 64}),
            "manifestDigest",
        ),
        (
            lambda value: value["artifacts"]["mesh"]["obj"].update({"relativePath": "/tmp/a.obj"}),
            "relativePath",
        ),
        (
            lambda value: value["artifacts"]["mesh"].update({"textures": []}),
            "textures",
        ),
        (
            lambda value: value["coordinates"].update({"horizontal": "EPSG:4326"}),
            "coordinates",
        ),
    ],
)
def test_odx_output_rejects_drift_or_ambient_paths(mutate, message: str) -> None:
    input_manifest = load(INPUT)
    output_manifest = copy.deepcopy(load(OUTPUT))
    mutate(output_manifest)

    with pytest.raises(ValueError, match=message):
        validate_odx_output_manifest(output_manifest, input_manifest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["checkpoint"].update({"contentDigest": "sha256:" + "0" * 64}),
        lambda value: value["producer"].update({"imageDigest": "sha256:" + "0" * 64}),
    ],
)
def test_checkpoint_rejects_digest_or_implementation_drift(mutate) -> None:
    input_manifest = load(INPUT)
    output_manifest = copy.deepcopy(load(OUTPUT))
    mutate(output_manifest)

    with pytest.raises(ValueError, match="checkpoint"):
        validate_odx_output_manifest(output_manifest, input_manifest)
    assert "checkpoint.contentDigest" in " ".join(load(SCHEMA)["x-semanticConstraints"])


def test_remesh_receives_only_validated_manifest_sources() -> None:
    sources = remesh_sources(load(OUTPUT), load(INPUT))

    assert set(sources) == {"inputManifestDigest", "mesh", "coordinates", "checkpointId"}
    assert sources["mesh"]["obj"]["relativePath"] == "odm_texturing/odm_textured_model_geo.obj"
    assert "command" not in json.dumps(sources)
    assert "/tmp" not in json.dumps(sources)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"command": ["odm"]}),
        lambda value: value["producer"].update({"imageDigest": "latest"}),
        lambda value: value["artifacts"]["mesh"]["obj"].update({"relativePath": "stage/../outside.obj"}),
        lambda value: value["artifacts"]["mesh"].update({"textures": []}),
    ],
)
def test_published_schema_rejects_untrusted_shape(mutate) -> None:
    output_manifest = copy.deepcopy(load(OUTPUT))
    mutate(output_manifest)

    assert schema_errors(output_manifest)


@pytest.mark.parametrize("path", ["foo//bar", "foo/./bar", "foo/.", "foo/"])
def test_schema_and_semantic_validator_reject_ambiguous_paths(path: str) -> None:
    input_manifest = load(INPUT)
    output_manifest = copy.deepcopy(load(OUTPUT))
    output_manifest["artifacts"]["mesh"]["obj"]["relativePath"] = path

    assert schema_errors(output_manifest)
    with pytest.raises(ValueError, match="relativePath"):
        validate_odx_output_manifest(output_manifest, input_manifest)


def test_semantic_validator_rejects_duplicate_artifact_identity() -> None:
    input_manifest = load(INPUT)
    output_manifest = copy.deepcopy(load(OUTPUT))
    output_manifest["artifacts"]["mesh"]["mtl"]["artifactId"] = output_manifest["artifacts"]["mesh"]["obj"][
        "artifactId"
    ]

    with pytest.raises(ValueError, match="artifactId must be unique"):
        validate_odx_output_manifest(output_manifest, input_manifest)
    assert "artifactId" in " ".join(load(SCHEMA)["x-semanticConstraints"])


def test_semantic_validator_classifies_oversized_coordinate_as_value_error() -> None:
    input_manifest = load(INPUT)
    output_manifest = copy.deepcopy(load(OUTPUT))
    output_manifest["coordinates"]["originOffsetMeters"][0] = 10**1000

    with pytest.raises(ValueError, match="finite number"):
        validate_odx_output_manifest(output_manifest, input_manifest)
