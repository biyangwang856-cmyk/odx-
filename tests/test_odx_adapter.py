from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from core.artifacts import safe_extract_tar
from core.contracts.odx_output import validate_odx_output_manifest
from core.validation.odx_output import validate_textured_mesh, verify_output_files
from experiments.registry import get_task
from experiments.tools.prepare_odx_input import prepare
from model_engine.common import event_artifact_ref
from model_engine.odx import (
    ODX_IMAGE,
    build_odx_command,
    build_odx_output_manifest,
    classify_adapter_error,
    create_odx_bundle,
)
from odx_fixtures import make_output

ROOT = Path(__file__).parents[1]
INPUT = ROOT / "tests/fixtures/contracts/photogrammetry-odx-v1.lglglg.json"


def input_manifest() -> dict[str, object]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def test_probe_input_is_bound_to_actual_image_bytes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"source image bytes")
    profile = ROOT / "app/experiments/specs/wp18a-odx-24.json"
    archive = tmp_path / "input.tar"
    prepare(images, profile, archive)
    with tarfile.open(archive) as source:
        manifest = json.load(source.extractfile("input-manifest.json"))
        assert manifest["images"]["fileCount"] == 1
        assert manifest["images"]["bytes"] == len(b"source image bytes")
        assert source.extractfile("images/one.jpg").read() == b"source image bytes"
    with pytest.raises(FileExistsError):
        prepare(images, profile, archive)


def test_odx_command_is_digest_pinned_and_allowlisted(tmp_path: Path) -> None:
    command = build_odx_command(tmp_path / "datasets", input_manifest())
    assert command[:2] == [sys.executable, "/code/run.py"]
    assert command[command.index("--end-with") + 1] == "odm_orthophoto"
    assert "--cog" in command
    assert "docker" not in command
    assert "--skip-orthophoto" not in command
    assert "--3d-tiles" not in command
    assert "--gltf" not in command
    assert "--feature-quality" in command and "ultra" in command
    assert "--rerun-all" not in command
    assert ODX_IMAGE.startswith("docker.io/webodm/odx@sha256:")
    assert get_task("adapter.photogrammetry-odx").category == "adapter"


def test_odx_command_rejects_unsupported_tone_mapping(tmp_path: Path) -> None:
    manifest = input_manifest()
    manifest["odxParameters"]["texturingToneMapping"] = "reinhard"
    with pytest.raises(ValueError, match="texturingToneMapping"):
        build_odx_command(tmp_path, manifest)


@pytest.mark.parametrize("mode", [None, "3d", "2.5d"])
def test_orthophoto_mode_preserves_full_3d_output(tmp_path: Path, mode: str | None) -> None:
    manifest = input_manifest()
    if mode is not None:
        manifest["odxParameters"]["orthophotoMode"] = mode
    command = build_odx_command(tmp_path, manifest)
    assert ("--use-3dmesh" in command) == (mode != "2.5d")
    assert "--skip-3dmodel" not in command
    assert "--fast-orthophoto" not in command


def test_mode_is_bound_to_output_identity(validated_output) -> None:
    root, original = validated_output
    manifest = input_manifest()
    manifest["odxParameters"]["orthophotoMode"] = "2.5d"
    changed = build_odx_output_manifest(root, manifest)
    validate_odx_output_manifest(changed, manifest)
    assert changed["parametersDigest"] != original["parametersDigest"]
    assert changed["outputId"] != original["outputId"]
    assert changed["artifacts"]["mesh"] == original["artifacts"]["mesh"]


@pytest.fixture
def validated_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    make_output(root)
    # Physical GDAL checks run in the pinned ODX integration environment.
    monkeypatch.setattr("model_engine.odx.validate_cog", lambda path, crs: {"crs": crs})
    monkeypatch.setattr("model_engine.odx.read_origin_offset", lambda path, crs: [215613, 2519070, 0])
    return root, build_odx_output_manifest(root, input_manifest())


def test_output_manifest_inventories_validated_shared_outputs(validated_output) -> None:
    root, manifest = validated_output
    validate_odx_output_manifest(manifest, input_manifest())
    verify_output_files(root, manifest)
    assert manifest["producer"]["imageDigest"] == ODX_IMAGE.split("@", 1)[1]
    assert set(manifest["artifacts"]) == {"mesh", "orthophoto"}
    assert manifest["coordinates"]["originOffsetMeters"] == [215613, 2519070, 0]
    assert manifest["checkpoint"]["stage"] == "odm_orthophoto"


def test_output_identity_includes_coordinate_offset(validated_output, monkeypatch) -> None:
    root, original = validated_output
    monkeypatch.setattr("model_engine.odx.read_origin_offset", lambda path, crs: [215614, 2519070, 0])
    changed = build_odx_output_manifest(root, input_manifest())
    assert changed["outputId"] != original["outputId"]
    assert changed["checkpoint"]["contentDigest"] != original["checkpoint"]["contentDigest"]


def test_bundle_contains_only_declared_shared_outputs(validated_output, tmp_path: Path) -> None:
    root, manifest = validated_output
    (root / "unused-work").write_bytes(b"not part of the delivery")
    input_path = tmp_path / "input-manifest.json"
    output_path = tmp_path / "odx-output-manifest.json"
    input_path.write_text(json.dumps(input_manifest()))
    output_path.write_text(json.dumps(manifest))
    bundle_path = tmp_path / "output.tar"
    create_odx_bundle(root, input_path, output_path, bundle_path)
    extracted = tmp_path / "extracted"
    safe_extract_tar(bundle_path, extracted)
    verify_output_files(extracted, manifest)
    assert not (extracted / "unused-work").exists()
    assert (extracted / "input-manifest.json").is_file()
    assert (extracted / "odx-output-manifest.json").is_file()


def test_shared_output_drift_is_rejected(validated_output) -> None:
    root, manifest = validated_output
    (root / manifest["artifacts"]["mesh"]["obj"]["relativePath"]).write_text("changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        verify_output_files(root, manifest)


def test_missing_texture_and_invalid_geometry_fail(tmp_path: Path) -> None:
    make_output(tmp_path)
    mesh = tmp_path / "odm_texturing"
    obj, mtl = mesh / "odm_textured_model_geo.obj", mesh / "odm_textured_model_geo.mtl"
    assert len(validate_textured_mesh(obj, mtl)) == 1
    (mesh / "material0000.jpg").unlink()
    with pytest.raises(ValueError, match="texture is missing"):
        validate_textured_mesh(obj, mtl)
    obj.write_text("v 0 0 0\n")
    with pytest.raises(ValueError, match="geometry"):
        validate_textured_mesh(obj, mtl)


def test_each_used_material_needs_a_texture(tmp_path: Path) -> None:
    make_output(tmp_path)
    mesh = tmp_path / "odm_texturing"
    obj, mtl = mesh / "odm_textured_model_geo.obj", mesh / "odm_textured_model_geo.mtl"
    with obj.open("a") as output:
        output.write("usemtl untextured\nf 1/1 2/2 3/3\n")
    with mtl.open("a") as output:
        output.write("newmtl untextured\n")
    with pytest.raises(ValueError, match="materials or textures"):
        validate_textured_mesh(obj, mtl)


def test_a_face_before_material_assignment_is_rejected(tmp_path: Path) -> None:
    make_output(tmp_path)
    mesh = tmp_path / "odm_texturing"
    obj, mtl = mesh / "odm_textured_model_geo.obj", mesh / "odm_textured_model_geo.mtl"
    obj.write_text(obj.read_text().replace("usemtl material0\n", "") + "usemtl material0\n")
    with pytest.raises(ValueError, match="face has no material"):
        validate_textured_mesh(obj, mtl)


def test_safe_extract_rejects_paths_and_links_outside_job_root(tmp_path: Path) -> None:
    for name, link in [("../outside", None), ("images/link.jpg", "/etc/passwd")]:
        archive = tmp_path / f"bad-{len(name)}.tar"
        with tarfile.open(archive, "w") as output:
            member = tarfile.TarInfo(name)
            if link is None:
                member.size = 1
                output.addfile(member, io.BytesIO(b"x"))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                output.addfile(member)
        with pytest.raises(ValueError):
            safe_extract_tar(archive, tmp_path / "extract")


def test_output_symlinks_and_operational_archive_errors_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "outside").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symbolic link"):
        from core.artifacts import tree_identity

        tree_identity(root)

    assert classify_adapter_error(tarfile.ReadError("bad archive"), "input") == ("adapter_failed", 1)
    assert classify_adapter_error(ValueError("bad manifest"), "input") == ("invalid_input", 2)


def test_event_artifact_uses_the_oneshot_wire_shape(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}")

    reference = event_artifact_ref(path, tmp_path, "application/json", "manifest")

    assert set(reference) == {
        "artifact_id",
        "relative_path",
        "content_digest",
        "bytes",
        "media_type",
    }
