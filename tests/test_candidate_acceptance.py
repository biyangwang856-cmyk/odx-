from __future__ import annotations

import json
from pathlib import Path

from core.validation.tilesets import audit_candidate, audit_filesystem
from experiments.quality.candidate_acceptance import select_default

SELECTION = Path(__file__).parents[1] / "app/experiments/specs/lglglg-candidate-selection-v1.json"


def write_tileset(root: Path, uri: str = "tile.glb") -> None:
    root.mkdir()
    (root / "tileset.json").write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "geometricError": 10,
                "root": {
                    "boundingVolume": {"sphere": [0, 0, 0, 1]},
                    "geometricError": 0,
                    "content": {"uri": uri},
                },
            }
        ),
        encoding="utf-8",
    )


def test_filesystem_audit_checks_every_content_reference(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    write_tileset(root)
    (root / "tile.glb").write_bytes(b"glTFpayload")

    report = audit_filesystem(root)

    assert report == {
        "passed": True,
        "tilesets": 1,
        "contentReferences": 1,
        "uniqueContents": 1,
        "extensions": {".glb": 1},
        "failures": [],
    }


def test_filesystem_audit_rejects_missing_and_escaping_content(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    write_tileset(missing, "missing.glb")
    escaping = tmp_path / "escaping"
    write_tileset(escaping, "../outside.glb")

    assert audit_filesystem(missing)["passed"] is False
    assert audit_filesystem(escaping)["passed"] is False


def test_http_audit_requires_range_cors_and_content_magic(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    write_tileset(root)
    (root / "tile.glb").write_bytes(b"glTFpayload")
    tileset_body = (root / "tileset.json").read_bytes()

    class Response:
        def __init__(self, status: int, body: bytes, headers: dict[str, str]):
            self.status = status
            self.body = body
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int | None = None) -> bytes:
            return self.body

    def opener(request, *, timeout: int):
        assert timeout == 10
        if request.full_url.endswith("tileset.json"):
            return Response(200, tileset_body, {"Access-Control-Allow-Origin": "*"})
        assert request.headers["Range"] == "bytes=0-3"
        return Response(
            206,
            b"glTF",
            {
                "Access-Control-Allow-Origin": "*",
                "Content-Range": "bytes 0-3/11",
            },
        )

    report = audit_candidate(
        "remesh-v55",
        root,
        "http://example.test/a1",
        "http://127.0.0.1:3000",
        opener=opener,
    )

    assert report["passed"] is True
    assert report["http"] == {
        "passed": True,
        "jsonRequests": 1,
        "rangeRequests": 1,
        "failures": [],
    }


def test_http_audit_rejects_a_structurally_invalid_tileset(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    write_tileset(root)
    (root / "tile.glb").write_bytes(b"glTFpayload")

    class Response:
        def __init__(self, status: int, body: bytes, headers: dict[str, str]):
            self.status = status
            self.body = body
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int | None = None) -> bytes:
            return self.body

    def opener(request, **_kwargs):
        if request.full_url.endswith("tileset.json"):
            return Response(200, b"{}", {"Access-Control-Allow-Origin": "*"})
        return Response(
            206,
            b"glTF",
            {
                "Access-Control-Allow-Origin": "*",
                "Content-Range": "bytes 0-3/11",
            },
        )

    report = audit_candidate(
        "remesh-v55",
        root,
        "http://example.test/a1",
        "http://127.0.0.1:3000",
        opener=opener,
    )

    assert report["filesystem"]["passed"] is True
    assert report["http"]["passed"] is False
    assert report["http"]["failures"] == [
        "HTTP tileset is invalid: tileset.json: tileset asset version is invalid",
        "HTTP tileset is invalid: tileset.json: tileset geometricError is invalid",
        "HTTP tileset is invalid: tileset.json: tileset root is missing",
    ]


def profile(candidate_id: str, triangles: int, memory: int, transfer: int, timed_out: bool) -> dict:
    return {
        "schemaVersion": "mesh-pair-profile.v1",
        "protocolDigest": "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
        "models": {"right": {"candidateId": candidate_id}},
        "viewerBinding": {"models": {"right": {"candidateId": candidate_id}}},
        "views": ["overview", "building", "texture", "edge"],
        "phases": [
            {
                "timedOut": timed_out,
                "final": {"right": {"visible": 205, "triangles": triangles}},
                "peaks": {"rightMemoryBytes": memory},
            }
        ],
        "resources": {"right": {"bytes": transfer}},
        "errors": [],
        "failedRequests": [],
    }


def test_selection_chooses_v55_with_a_known_timeout_boundary() -> None:
    plan = json.loads(SELECTION.read_text(encoding="utf-8"))
    profiles = {
        "remesh-v55": profile("remesh-v55", 6_211_035, 767_565_024, 163_515_669, True),
    }
    checks = {"remesh-v55": {"passed": True}}

    decision = select_default(
        plan,
        profiles,
        publication_bytes={"remesh-v55": 107_905_788},
        structure=checks,
        http=checks,
    )

    assert decision["defaultCandidateId"] == "remesh-v55"
    assert decision["selectionStatus"] == "selected-with-known-timeout-boundary"
    assert decision["candidates"]["remesh-v55"]["meetsSelectionBudget"] is True
    assert decision["candidates"]["remesh-v55"]["profilePassed"] is False


def test_selection_rejects_v55_with_console_errors() -> None:
    plan = json.loads(SELECTION.read_text(encoding="utf-8"))
    profiles = {
        "remesh-v55": profile("remesh-v55", 6_211_035, 767_565_024, 163_515_669, True),
    }
    profiles["remesh-v55"]["errors"] = ["viewer failed"]
    checks = {"remesh-v55": {"passed": True}}

    decision = select_default(
        plan,
        profiles,
        publication_bytes={"remesh-v55": 107_905_788},
        structure=checks,
        http=checks,
    )

    assert decision["defaultCandidateId"] is None
    assert decision["candidates"]["remesh-v55"]["budgetResults"]["consoleErrors"] is False
