from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from core.contracts.photogrammetry_odx import (
    photogrammetry_odx_input_digest,
    validate_photogrammetry_odx_input,
)
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "app/core/contracts/schemas/photogrammetry-odx-v1.schema.json"
EXAMPLE = ROOT / "tests/fixtures/contracts/photogrammetry-odx-v1.lglglg.json"


def load_example() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def schema_errors(manifest: dict[str, object]) -> list[object]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return list(validator.iter_errors(manifest))


def set_ground_control_media_type(manifest: dict[str, object], media_type: str) -> None:
    manifest["groundControl"] = {
        "artifactId": "artifact:gcp:lglglg-v1",
        "contentDigest": "sha256:" + "a" * 64,
        "bytes": 128,
        "mediaType": media_type,
    }


def test_lglglg_odx_input_is_a_complete_versioned_contract() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    manifest = load_example()

    validate_photogrammetry_odx_input(manifest)
    assert schema_errors(manifest) == []
    assert schema["$id"] == "https://surveying.local/schemas/photogrammetry-odx-v1.schema.json"
    assert manifest["schemaVersion"] == "photogrammetry.odx.v1"
    assert manifest["inputId"] == "lglglg-acquisition-20260814"
    assert manifest["images"]["contentDigest"] == (
        "sha256:17d7a4f7f7e7da0a64a55894edabfb16bfbeffbd2184e40fa0eefddcae60107b"
    )
    assert set(manifest["odxParameters"]) == {
        "featureQuality",
        "matcherNeighbors",
        "meshOctreeDepth",
        "meshSize",
        "pcQuality",
        "texturingToneMapping",
    }
    assert photogrammetry_odx_input_digest(manifest) == (
        "sha256:32b39644782124dd9fe3438030719d63565e3eb4361e0ac69319d927dd823355"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"command": ["odm", "--unsafe"]}), "unknown fields"),
        (
            lambda value: value["images"].update({"contentDigest": "sha256:not-hex"}),
            "contentDigest",
        ),
        (lambda value: value["odxParameters"].update({"rerunAll": True}), "unknown fields"),
        (lambda value: value["resourceBudget"].update({"maxImages": 0}), "maxImages"),
        (lambda value: value["crs"].update({"horizontal": "32650"}), "horizontal"),
        (lambda value: value["crs"].update({"horizontal": "EPSG:１２"}), "horizontal"),
        (
            lambda value: value["acquisition"].update({"capturedAt": "2026-08-14 12:00:00+00:00"}),
            "capturedAt",
        ),
        (lambda value: set_ground_control_media_type(value, "textcsv"), "mediaType"),
        (lambda value: set_ground_control_media_type(value, "text/"), "mediaType"),
        (lambda value: set_ground_control_media_type(value, "/csv"), "mediaType"),
        (lambda value: set_ground_control_media_type(value, "text//csv"), "mediaType"),
        (lambda value: set_ground_control_media_type(value, "text/csv\n"), "mediaType"),
        (
            lambda value: value["acquisition"].update({"capturedAt": "2026-08-14T12:00:00Z\n"}),
            "capturedAt",
        ),
    ],
)
def test_odx_input_rejects_ambiguous_or_unsafe_values(mutate, message: str) -> None:
    manifest = copy.deepcopy(load_example())
    mutate(manifest)

    with pytest.raises(ValueError, match=message):
        validate_photogrammetry_odx_input(manifest)
    assert schema_errors(manifest), "published schema accepted a rejected manifest"


def test_optional_inputs_remain_content_addressed() -> None:
    manifest = load_example()
    manifest["groundControl"] = {
        "artifactId": "artifact:gcp:lglglg-v1",
        "contentDigest": "sha256:" + "a" * 64,
        "bytes": 128,
        "mediaType": "text/csv",
    }
    manifest["geoFile"] = {
        "artifactId": "artifact:geo:lglglg-v1",
        "contentDigest": "sha256:" + "b" * 64,
        "bytes": 256,
        "mediaType": "text/plain",
    }
    manifest["imageGroups"] = {
        "artifactId": "artifact:image-groups:lglglg-v1",
        "contentDigest": "sha256:" + "c" * 64,
        "bytes": 64,
        "mediaType": "application/json",
    }

    validate_photogrammetry_odx_input(manifest)
    assert schema_errors(manifest) == []


@pytest.mark.parametrize("mode", ["3d", "2.5d"])
def test_orthophoto_mode_matches_published_schema(mode: str) -> None:
    manifest = load_example()
    manifest["odxParameters"]["orthophotoMode"] = mode
    validate_photogrammetry_odx_input(manifest)
    assert schema_errors(manifest) == []


@pytest.mark.parametrize("mode", ["2d", "2.5D", True, 2.5, None])
def test_unsupported_orthophoto_mode_is_rejected(mode: object) -> None:
    manifest = load_example()
    manifest["odxParameters"]["orthophotoMode"] = mode
    with pytest.raises(ValueError, match="orthophotoMode"):
        validate_photogrammetry_odx_input(manifest)
    assert schema_errors(manifest)
