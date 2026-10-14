from __future__ import annotations

import json
from http.client import IncompleteRead
from pathlib import Path
from urllib.request import Request

import pytest
from experiments.analysis.candidate_inventory import inventory_candidates

PROTOCOL = Path(__file__).parents[1] / "app/experiments/specs/lglglg-candidate-baseline-v1.json"


def test_inventory_records_available_and_missing_candidates(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    terra_root = tmp_path / "terra"
    terra_root.mkdir()
    (terra_root / "tileset.json").write_text(
        '{"asset":{"version":"1.1"},"geometricError":1,"root":{}}\n', encoding="utf-8"
    )

    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": str(terra_root)},
    )

    assert report["schemaVersion"] == "mesh-candidate-inventory.v1"
    assert report["sourceImages"] == protocol["dataset"]["images"]
    assert report["candidates"][0] == {
        "id": "terra-5.1.1",
        "label": "DJI Terra 5.1.1",
        "kind": "comparison-baseline",
        "productionEligible": False,
        "revisionKind": "engine",
        "revision": "5.1.1",
        "inputSource": "surveying-scene/LGLGLG/terra_b3dms",
        "buildSource": None,
        "expectedPublication": {
            "algorithm": "sha256-tree-v1",
            "files": 1794,
            "bytes": 558170657,
            "digest": "sha256:23c4f0d6115afb36d850ebfdfc20e03340b8776d2bfed4eb0a374032f4251741",
        },
        "rootEnvironment": "MESH_RESEARCH_TERRA_ROOT",
        "root": str(terra_root),
        "availability": "publication-mismatch",
        "tilesetDigest": "sha256:82c26daaa1b7f056334251da0fa6539b9f3baff6eb883eecc4dfb4e1a0e5bd78",
        "observedPublication": {
            "algorithm": "sha256-tree-v1",
            "files": 1,
            "bytes": 57,
            "digest": "sha256:2cbd6de90a0366def4de1b25e6151c682a21c67f461ff9f217fdedd7b804d4ea",
        },
        "evidenceGaps": [
            "publication-identity",
            "structure",
            "browser-profile",
            "screenshots",
            "http-range-cors",
        ],
    }
    assert report["candidates"][1]["id"] == "odx-3.8.3-eb9fd56"
    assert report["candidates"][1]["availability"] == "missing-root-environment"
    assert report["candidates"][1]["revision"] == ("eb9fd56ac430fc3735401d88aaef9945ebd00e14")
    assert report["productionMeshCandidateIds"] == ["remesh-v55"]
    assert report["recommendedProductionMeshId"] == "remesh-v55"
    assert report["allCandidatesVerified"] is False


def test_inventory_distinguishes_missing_root_and_tileset(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    report = inventory_candidates(
        protocol,
        {
            "MESH_RESEARCH_TERRA_ROOT": str(tmp_path / "missing"),
            "MESH_RESEARCH_ODX_ROOT": str(empty_root),
        },
    )

    assert report["candidates"][0]["availability"] == "missing-root"
    assert report["candidates"][1]["availability"] == "missing-tileset"


def test_inventory_reads_a_candidate_from_an_http_viewer(monkeypatch) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    body = b'{"asset":{"version":"1.1"},"geometricError":1,"root":{}}\n'

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            return body

    def fake_open(request: Request, *, timeout: int):
        assert request.full_url == "http://127.0.0.1:18147/terra/tileset.json"
        assert timeout == 10
        return Response()

    monkeypatch.setattr("experiments.analysis.candidate_inventory.urlopen", fake_open)

    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": "http://127.0.0.1:18147/terra"},
    )

    assert report["candidates"][0]["root"] == "http://127.0.0.1:18147/terra"
    assert report["candidates"][0]["availability"] == "identity-unverified"
    assert report["candidates"][0]["tilesetDigest"] == (
        "sha256:82c26daaa1b7f056334251da0fa6539b9f3baff6eb883eecc4dfb4e1a0e5bd78"
    )


def test_inventory_preserves_truncated_http_failures(monkeypatch) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            raise IncompleteRead(b"partial")

    monkeypatch.setattr(
        "experiments.analysis.candidate_inventory.urlopen", lambda *_args, **_kwargs: Response()
    )
    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": "http://127.0.0.1:18147/terra"},
    )

    assert report["candidates"][0]["availability"] == "unreachable"


def test_inventory_preserves_malformed_url_failures() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_ODX_ROOT": "http://[invalid"},
    )

    assert report["candidates"][1]["availability"] == "unreachable"


def test_inventory_rejects_json_that_is_not_a_root_tileset(monkeypatch) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            return b"{}"

    monkeypatch.setattr(
        "experiments.analysis.candidate_inventory.urlopen", lambda *_args, **_kwargs: Response()
    )
    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": "http://127.0.0.1:18147/terra"},
    )

    assert report["candidates"][0]["availability"] == "invalid-tileset"
    assert report["allCandidatesVerified"] is False


@pytest.mark.parametrize(
    ("version", "geometric_error"),
    [
        ("1.1", "true"),
        (" ", "1"),
        ("1.1", "NaN"),
        ("1.1", "Infinity"),
        ("1.1", "-Infinity"),
        ("1.1", "1" + "0" * 1000),
    ],
)
def test_inventory_rejects_invalid_root_tileset_numbers(
    monkeypatch, version: str, geometric_error: str
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    body = (f'{{"asset":{{"version":"{version}"}},"geometricError":{geometric_error},"root":{{}}}}').encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            return body

    monkeypatch.setattr(
        "experiments.analysis.candidate_inventory.urlopen", lambda *_args, **_kwargs: Response()
    )
    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": "http://127.0.0.1:18147/terra"},
    )

    assert report["candidates"][0]["availability"] == "invalid-tileset"


def test_inventory_preserves_local_io_failures(tmp_path: Path, monkeypatch) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    root = tmp_path / "terra"
    root.mkdir()
    tileset = root / "tileset.json"
    tileset.write_text('{"asset":{"version":"1.1"},"geometricError":1,"root":{}}\n', encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_tileset_read(path: Path) -> bytes:
        if path == tileset:
            raise PermissionError("denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_tileset_read)

    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": str(root)},
    )

    assert report["candidates"][0]["availability"] == "unreachable"


def test_inventory_preserves_local_probe_failures(tmp_path: Path, monkeypatch) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    root = tmp_path / "terra"
    root.mkdir()
    original_is_dir = Path.is_dir

    def fail_root_probe(path: Path) -> bool:
        if path == root:
            raise PermissionError("denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_root_probe)
    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": str(root)},
    )

    assert report["candidates"][0]["availability"] == "unreachable"


def test_inventory_preserves_symlink_loop_failures(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second, target_is_directory=True)
    second.symlink_to(first, target_is_directory=True)

    report = inventory_candidates(
        protocol,
        {"MESH_RESEARCH_TERRA_ROOT": str(first)},
    )

    assert report["candidates"][0]["availability"] == "unreachable"
