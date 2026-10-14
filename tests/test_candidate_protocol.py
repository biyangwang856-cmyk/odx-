from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.artifacts import image_tree_identity
from experiments.analysis.candidate_protocol import validate_protocol, verify_protocol

PROTOCOL = Path(__file__).parents[1] / "app/experiments/specs/lglglg-candidate-baseline-v1.json"


def test_repository_protocol_freezes_the_phase_zero_comparison() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schemaVersion"] == "mesh-candidate-protocol.v1"
    assert protocol["dataset"]["images"] == {
        "algorithm": "sha256-tree-v1",
        "count": 546,
        "bytes": 4_444_096_141,
        "digest": "sha256:17d7a4f7f7e7da0a64a55894edabfb16bfbeffbd2184e40fa0eefddcae60107b",
    }
    assert protocol["browser"]["viewport"] == {"width": 1400, "height": 900}
    assert protocol["browser"]["navigationTimeoutMs"] == 90000
    assert protocol["browser"]["views"] == ["overview", "building", "texture", "edge"]
    assert protocol["browser"]["anchor"] == {
        "candidateId": "remesh-v55",
        "publicationDigest": "sha256:d76db8931e5fd6aaf3cf17240c247751d32389681b5ac552eb5f000fab506d6c",
        "tilesetDigest": "sha256:7ee09e40fea72d70ae70b709c70b8e957c9f7d92c1175e4e6ccb8e6ec18278f2",
        "derivation": "Cesium3DTileset.boundingSphere",
    }
    assert [(pair["leftId"], pair["rightId"]) for pair in protocol["browser"]["profilePairs"]] == [
        ("terra-5.1.1", "remesh-v55"),
        ("odx-3.8.3-eb9fd56", "remesh-v55"),
    ]
    assert protocol["browser"]["profilePairs"][1]["leftDisplayLabel"] == "ODX 3.8.3"
    assert [candidate["id"] for candidate in protocol["candidates"]] == [
        "terra-5.1.1",
        "odx-3.8.3-eb9fd56",
        "remesh-v55",
    ]
    assert protocol["roles"] == {
        "comparisonBaselineId": "terra-5.1.1",
        "photogrammetrySourceId": "odx-3.8.3-eb9fd56",
        "productionMeshCandidateIds": ["remesh-v55"],
        "recommendedProductionMeshId": "remesh-v55",
    }
    assert [candidate["kind"] for candidate in protocol["candidates"]] == [
        "comparison-baseline",
        "photogrammetry-source",
        "production-mesh",
    ]
    assert [candidate["productionEligible"] for candidate in protocol["candidates"]] == [
        False,
        False,
        True,
    ]
    assert protocol["candidates"][1]["revision"] == ("eb9fd56ac430fc3735401d88aaef9945ebd00e14")


def test_image_tree_identity_matches_the_documented_algorithm(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "a.jpg").write_bytes(b"first")
    (image_root / "b.jpg").write_bytes(b"second")

    assert image_tree_identity(image_root) == {
        "algorithm": "sha256-tree-v1",
        "count": 2,
        "bytes": 11,
        "digest": "sha256:58f907f6d7a956f97d8fdc6a56fb0e76a3549e94db2ae8f844993a987a236e4d",
    }


def test_verify_protocol_rejects_input_identity_drift(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "unexpected.jpg").write_bytes(b"different acquisition")

    with pytest.raises(ValueError, match="image identity mismatch"):
        verify_protocol(protocol, image_root)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dataset", "coordinates", "scale"), 2.0),
        (("browser", "browserRevision"), "different"),
        (("browser", "cameraRules", "overview", "heading"), 26),
        (("candidates", 0, "revision"), "different"),
        (("candidates", 1, "rootEnvironment"), "DIFFERENT_ROOT"),
        (("commands", "profile"), "different command"),
    ],
)
def test_validate_protocol_rejects_any_frozen_contract_drift(
    path: tuple[str | int, ...], value: object
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    target = protocol
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="protocol digest mismatch"):
        validate_protocol(protocol)


def test_image_tree_identity_rejects_broken_symbolic_links(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "a.jpg").write_bytes(b"first")
    (image_root / "broken.jpg").symlink_to("missing.jpg")

    with pytest.raises(ValueError, match="symbolic link"):
        image_tree_identity(image_root)


def test_image_tree_identity_rejects_a_symbolic_link_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-images"
    real_root.mkdir()
    (real_root / "a.jpg").write_bytes(b"first")
    linked_root = tmp_path / "linked-images"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="image root is a symbolic link"):
        image_tree_identity(linked_root)
