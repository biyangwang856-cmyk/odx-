from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from core.artifacts import sha256_file
from core.contracts.staging import validate_staging_manifest
from core.pipelines.source_leaf_tiler import crop_texture
from experiments.registry import get_task
from jsonschema import Draft202012Validator
from model_engine.common import AdapterEnvironment, EventEmitter
from model_engine.odx import build_odx_output_manifest, create_odx_bundle
from model_engine.remesh import (
    build_remesh_command,
    build_staging_manifest,
    create_staging_bundle,
    parse_config,
)
from odx_fixtures import make_output
from PIL import Image

ROOT = Path(__file__).parents[1]
ODX = ROOT / "tests/fixtures/contracts/odx-output-manifest-v2.fixture.json"
CATALOG = ROOT / "configs/mesh-candidates.json"
SCHEMA = ROOT / "app/core/contracts/schemas/mesh-staging-manifest-v2.schema.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, object]:
    return {
        "odxManifestDigest": "sha256:" + "a" * 64,
        "candidateId": "mesh:remesh:v55-runtime",
        "featuresPerTile": 64,
        "maxOutputBytes": 10000,
    }


def write_build_report(root: Path) -> None:
    toolchain = {
        "basisUniversalVersion": "1.16.3",
        "basisUniversalGitCommit": "1531cfaf9ed5232248a0a45736686a849ca3befc",
        "basisUniversalSourceSha256": ("b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac"),
        "basisUniversalBinarySha256": "b" * 64,
        "policySha256": "c" * 64,
    }
    report = {
        "gpuTextures": {
            "codec": "etc1s",
            "geometryAndUvBytesPreserved": True,
            "models": [{"path": "tiles/tile.glb", "toolchain": toolchain}],
        }
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["reportPayloadDigest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    (root / "build-report.json").write_text(json.dumps(report), encoding="utf-8")


def write_tiles(root: Path) -> None:
    (root / "tiles").mkdir(parents=True)
    (root / "tiles/tile.glb").write_bytes(b"glTFfixture")
    (root / "tileset.json").write_text(
        json.dumps(
            {
                "asset": {"version": "1.1"},
                "root": {
                    "geometricError": 0,
                    "boundingVolume": {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
                    "content": {"uri": "tiles/tile.glb"},
                },
            }
        )
    )
    write_build_report(root)


def test_remesh_config_and_command_use_only_registry_contract(tmp_path: Path) -> None:
    parsed = parse_config(config())
    command = build_remesh_command(
        tmp_path / "model.obj",
        tmp_path / "tiles",
        load(ODX),
        parsed,
    )

    assert command[1:3] == ["-m", "core.pipelines.source_leaf_tiler"]
    assert "--input-obj" in command
    proj_index = command.index("--proj-data")
    assert command[proj_index + 1] == str(Path(sys.prefix) / "share/proj")
    assert "EPSG:32650" in command
    crop_index = command.index("--crop-textures-dir")
    assert command[crop_index + 1] == str(tmp_path / "texture-crops")
    size_index = command.index("--max-texture-size")
    assert command[size_index + 1] == "8192"
    assert get_task("adapter.remesh-3d-tiles").category == "adapter"


def test_remesh_config_rejects_algorithm_internal_resume() -> None:
    value = config()
    value["resumeFrom"] = "checkpoint:mutable:odx-workdir"

    with pytest.raises(ValueError, match="allowlist"):
        parse_config(value)


def test_texture_crop_caps_browser_dimension(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (16, 8), "red").save(source)

    result = crop_texture(
        {
            "source": source,
            "output": output,
            "cropBox": [0, 0, 16, 8],
            "maxTextureSize": 4,
        }
    )

    with Image.open(output) as image:
        assert image.size == (4, 2)
    assert result["outputSize"] == [4, 2]


def test_staging_manifest_requires_complete_references_and_budget(tmp_path: Path) -> None:
    write_tiles(tmp_path)
    manifest = build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))

    validate_staging_manifest(manifest, load(CATALOG))
    assert list(Draft202012Validator(load(SCHEMA)).iter_errors(manifest)) == []
    assert manifest["candidateId"] == "mesh:remesh:v55-runtime"
    assert manifest["validation"]["referencedContents"] == 1
    assert manifest["validation"]["missingContents"] == 0
    assert [item["path"] for item in manifest["files"]] == [
        "build-report.json",
        "tiles/tile.glb",
        "tileset.json",
    ]
    assert manifest["schemaVersion"] == "mesh-staging-manifest.v2"
    assert manifest["pipeline"]["basisUniversal"]["version"] == "1.16.3"
    assert sum(item["size_bytes"] for item in manifest["files"]) == manifest["validation"]["outputBytes"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert (
        next(item for item in manifest["files"] if item["path"] == "tileset.json")["content_type"]
        == "application/json"
    )

    (tmp_path / "tiles/tile.glb").unlink()
    with pytest.raises(FileNotFoundError):
        build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))


def test_staging_bundle_keeps_tileset_at_root(tmp_path: Path) -> None:
    tiles = tmp_path / "tiles"
    write_tiles(tiles)
    manifest_path = tmp_path / "mesh-staging-manifest.json"
    manifest = build_staging_manifest(tiles, load(ODX), config(), load(CATALOG))
    manifest_path.write_text(json.dumps(manifest))
    bundle_path = tmp_path / "mesh-staging.tar"

    create_staging_bundle(tiles, manifest_path, bundle_path)

    with tarfile.open(bundle_path) as bundle:
        names = set(bundle.getnames())
    assert names == {item["path"] for item in manifest["files"]} | {"mesh-staging-manifest.json"}
    assert "tiles/tiles/tile.glb" not in names
    assert "tiles/tile.glb" in names


def test_staging_bundle_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    tiles = tmp_path / "tiles"
    write_tiles(tiles)
    manifest = build_staging_manifest(tiles, load(ODX), config(), load(CATALOG))
    manifest_path = tmp_path / "mesh-staging-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    outside = tmp_path / "outside"
    (outside / "tile.glb").parent.mkdir(parents=True)
    (outside / "tile.glb").write_bytes((tiles / "tiles/tile.glb").read_bytes())
    (tiles / "tiles/tile.glb").unlink()
    (tiles / "tiles").rmdir()
    (tiles / "tiles").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="changed before archive"):
        create_staging_bundle(tiles, manifest_path, tmp_path / "bundle.tar")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.update({"candidateId": "remesh-v38"}),
        lambda value: value.update({"featuresPerTile": 0}),
        lambda value: value.update({"maxOutputBytes": 0}),
    ],
)
def test_remesh_config_rejects_unknown_alias_and_nonpositive_budget(mutate) -> None:
    value = copy.deepcopy(config())
    mutate(value)
    with pytest.raises(ValueError):
        parse_config(value)


@pytest.mark.parametrize("value", [None, [], [dict()]])
def test_remesh_config_rejects_non_objects(value: object) -> None:
    with pytest.raises(ValueError, match="object"):
        parse_config(value)


def test_staging_contract_rejects_boolean_counts_and_bad_coordinates(tmp_path: Path) -> None:
    write_tiles(tmp_path)
    manifest = build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))
    manifest["validation"]["missingContents"] = False
    with pytest.raises(ValueError, match="missingContents"):
        validate_staging_manifest(manifest, load(CATALOG))

    manifest = build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))
    manifest["coordinates"]["originOffsetMeters"][0] = "bad"
    with pytest.raises(ValueError, match="finite number"):
        validate_staging_manifest(manifest, load(CATALOG))


def test_staging_manifest_validates_multi_content_tiles(tmp_path: Path) -> None:
    write_tiles(tmp_path)
    (tmp_path / "tiles/tile-2.glb").write_bytes(b"glTFsecond")
    document = json.loads((tmp_path / "tileset.json").read_text())
    document["root"].pop("content")
    document["root"]["contents"] = [
        {"uri": "tiles/tile.glb"},
        {"uri": "tiles/tile-2.glb"},
    ]
    (tmp_path / "tileset.json").write_text(json.dumps(document))

    manifest = build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))

    assert manifest["validation"]["referencedContents"] == 2

    document["root"]["content"] = {"uri": "tiles/tile.glb"}
    (tmp_path / "tileset.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="both content and contents"):
        build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))


def test_staging_without_output_byte_ceiling(tmp_path: Path) -> None:
    write_tiles(tmp_path)
    unlimited = {**config(), "maxOutputBytes": None}
    assert parse_config(unlimited)["maxOutputBytes"] is None
    manifest = build_staging_manifest(tmp_path, load(ODX), unlimited, load(CATALOG))
    assert manifest["validation"]["maxOutputBytes"] is None
    assert manifest["validation"]["outputBytes"] > 0
    validate_staging_manifest(manifest, load(CATALOG))
    assert list(Draft202012Validator(load(SCHEMA)).iter_errors(manifest)) == []
    with pytest.raises(ValueError, match="exceeds maxOutputBytes"):
        build_staging_manifest(tmp_path, load(ODX), {**config(), "maxOutputBytes": 1}, load(CATALOG))


def test_staging_semantics_match_json_schema_integral_numbers(tmp_path: Path) -> None:
    write_tiles(tmp_path)
    manifest = build_staging_manifest(tmp_path, load(ODX), config(), load(CATALOG))
    manifest["tileset"]["contentTree"]["files"] = float(manifest["tileset"]["contentTree"]["files"])
    manifest["tileset"]["contentTree"]["bytes"] = float(manifest["tileset"]["contentTree"]["bytes"])
    for field in ("referencedContents", "missingContents", "outputBytes", "maxOutputBytes"):
        manifest["validation"][field] = float(manifest["validation"][field])

    validate_staging_manifest(manifest, load(CATALOG))
    assert list(Draft202012Validator(load(SCHEMA)).iter_errors(manifest)) == []


def test_remesh_main_runs_digest_to_staging_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_manifest = load(ROOT / "tests/fixtures/contracts/photogrammetry-odx-v1.lglglg.json")
    project = tmp_path / "odx-project"
    make_output(project)
    monkeypatch.setattr("model_engine.odx.validate_cog", lambda path, crs: {"crs": crs})
    monkeypatch.setattr("model_engine.odx.read_origin_offset", lambda path, crs: [215613, 2519070, 0])
    input_path = tmp_path / "input-manifest.json"
    input_path.write_text(json.dumps(input_manifest))
    odx_manifest_path = tmp_path / "odx-output-manifest.json"
    odx_manifest = build_odx_output_manifest(project, input_manifest)
    odx_manifest_path.write_text(json.dumps(odx_manifest, indent=2) + "\n")
    bundle_path = tmp_path / "odx-output.tar"
    create_odx_bundle(project, input_path, odx_manifest_path, bundle_path)
    job_root = tmp_path / "job"
    job_root.mkdir()
    task_config = config()
    task_config["odxManifestDigest"] = sha256_file(odx_manifest_path)
    environment = AdapterEnvironment(
        attempt_id="attempt-remesh-test",
        input_path=bundle_path,
        job_root=job_root,
        task_name="mesh.remesh-3d-tiles",
        task_revision="mesh.remesh-3d-tiles.v2",
        task_config=task_config,
    )
    stream = io.BytesIO()

    invoked = []

    def fake_run(command, **_kwargs):
        invoked.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        write_tiles(output)
        return SimpleNamespace(returncode=0)

    import model_engine.remesh as remesh

    monkeypatch.setattr(remesh, "AdapterEnvironment", SimpleNamespace(load=lambda: environment))
    monkeypatch.setattr(
        remesh,
        "EventEmitter",
        lambda value, adapter_id: EventEmitter(value, adapter_id, stream),
    )
    monkeypatch.setattr(remesh.subprocess, "run", fake_run)
    monkeypatch.setattr(remesh, "resolve_gltf_transform", lambda: "/tools/gltf-transform")
    monkeypatch.setattr(remesh, "resolve_node", lambda: "/tools/node")
    monkeypatch.setattr(remesh, "resolve_basisu", lambda: "/tools/basisu")
    monkeypatch.setattr(remesh, "resolve_basisu_provenance", lambda: "/tools/basisu.json")
    monkeypatch.setattr(remesh, "resolve_sharp_module", lambda: "/tools/sharp")

    def fake_build_publication(**arguments):
        write_tiles(arguments["output"])
        return {}

    monkeypatch.setattr(remesh, "build_publication", fake_build_publication)

    remesh.main()

    assert (job_root / "artifacts/mesh-staging-manifest.json").is_file()
    assert (job_root / "artifacts/mesh-staging.tar").is_file()
    events = []
    stream.seek(0)
    while header_size := stream.read(4):
        header = json.loads(stream.read(struct.unpack(">I", header_size)[0]))
        assert struct.unpack(">Q", stream.read(8))[0] == 0
        events.append(header)
    assert [event["event"] for event in events] == [
        "ready",
        "stage",
        "stage",
        "stage",
        "artifact-candidate",
        "artifact-candidate",
        "result",
    ]
    assert events[1]["checkpoint_id"] == odx_manifest["checkpoint"]["checkpointId"]
    assert events[3]["checkpoint_id"].startswith("checkpoint:staging:sha256:")
    result_path = job_root / events[-1]["manifest"]["relative_path"]
    assert events[-1]["manifest"]["content_digest"] == sha256_file(result_path)
    assert invoked[0][1:3] == ["-m", "core.pipelines.source_leaf_tiler"]
    input_obj = Path(invoked[0][invoked[0].index("--input-obj") + 1])
    assert input_obj.is_relative_to(job_root / "work/remesh-adapter/input")


def test_remesh_main_rejects_drifted_bundle_before_tiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_manifest = load(ROOT / "tests/fixtures/contracts/photogrammetry-odx-v1.lglglg.json")
    project = tmp_path / "odx-project"
    make_output(project)
    monkeypatch.setattr("model_engine.odx.validate_cog", lambda path, crs: {"crs": crs})
    monkeypatch.setattr("model_engine.odx.read_origin_offset", lambda path, crs: [215613, 2519070, 0])
    input_path = tmp_path / "input-manifest.json"
    input_path.write_text(json.dumps(input_manifest))
    odx_manifest_path = tmp_path / "odx-output-manifest.json"
    odx_manifest = build_odx_output_manifest(project, input_manifest)
    odx_manifest_path.write_text(json.dumps(odx_manifest, indent=2) + "\n")
    (project / odx_manifest["artifacts"]["mesh"]["obj"]["relativePath"]).write_text("changed")
    bundle_path = tmp_path / "drifted-odx-output.tar"
    create_odx_bundle(project, input_path, odx_manifest_path, bundle_path)
    job_root = tmp_path / "job"
    job_root.mkdir()
    task_config = config()
    task_config["odxManifestDigest"] = sha256_file(odx_manifest_path)
    environment = AdapterEnvironment(
        attempt_id="attempt-remesh-drift",
        input_path=bundle_path,
        job_root=job_root,
        task_name="mesh.remesh-3d-tiles",
        task_revision="mesh.remesh-3d-tiles.v2",
        task_config=task_config,
    )
    stream = io.BytesIO()

    import model_engine.remesh as remesh

    monkeypatch.setattr(remesh, "AdapterEnvironment", SimpleNamespace(load=lambda: environment))
    monkeypatch.setattr(
        remesh,
        "EventEmitter",
        lambda value, adapter_id: EventEmitter(value, adapter_id, stream),
    )
    monkeypatch.setattr(
        remesh.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("tiler ran before bundle bytes were verified"),
    )

    with pytest.raises(SystemExit) as stopped:
        remesh.main()

    assert stopped.value.code == 2
    stream.seek(0)
    events = []
    while header_size := stream.read(4):
        events.append(json.loads(stream.read(struct.unpack(">I", header_size)[0])))
        assert struct.unpack(">Q", stream.read(8))[0] == 0
    assert events[-1]["error"]["code"] == "invalid_input"
    assert "bytes do not match" in events[-1]["error"]["message"]
