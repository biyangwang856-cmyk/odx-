from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from core.artifacts import sha256_file, tree_identity
from core.contracts.staging import validate_staging_manifest
from core.validation.staging_gate import validate_staging_publication
from core.validation.staging_publication import publish_staging

ROOT = Path(__file__).parents[1]
CATALOG = json.loads((ROOT / "configs/mesh-candidates.json").read_text())


def write_glb(
    path: Path,
    *,
    textured: bool = True,
    triangles: int = 1,
    stored_triangles: int | None = None,
    image: bytes = b"\x89PNG\r\n\x1a\n",
    image_mime_type: str = "image/png",
    image_view_length: int | None = None,
    trailing: bytes = b"",
) -> None:
    stored_triangles = triangles if stored_triangles is None else stored_triangles
    indices = struct.pack(f"<{stored_triangles * 3}I", *range(stored_triangles * 3))
    positions = b"\x00" * (stored_triangles * 3 * 12)
    image_offset = len(indices) + len(positions)
    binary = indices + positions + image + trailing
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(indices)},
            {"buffer": 0, "byteOffset": len(indices), "byteLength": len(positions)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "count": triangles * 3,
                "componentType": 5125,
                "type": "SCALAR",
            }
        ],
        "meshes": [{"primitives": [{"mode": 4, "indices": 0, "attributes": {"POSITION": 1}}]}],
        "materials": [],
        "textures": [],
        "images": [],
    }
    document["accessors"].append(
        {
            "bufferView": 1,
            "count": stored_triangles * 3,
            "componentType": 5126,
            "type": "VEC3",
        }
    )
    if textured:
        image_view_length = len(image) if image_view_length is None else image_view_length
        document["bufferViews"].append(
            {"buffer": 0, "byteOffset": image_offset, "byteLength": image_view_length}
        )
        document["images"] = [{"bufferView": 2, "mimeType": image_mime_type}]
        document["textures"] = [{"source": 0}]
        document["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def glb_document(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    binary_header = 20 + json_length
    binary_length = struct.unpack_from("<I", data, binary_header)[0]
    return (
        json.loads(data[20:binary_header].decode()),
        data[binary_header + 8 : binary_header + 8 + binary_length],
    )


def write_glb_document(path: Path, document: dict[str, object], binary: bytes) -> None:
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def refresh_manifest(root: Path, manifest: dict[str, object]) -> None:
    tileset_path = root / "tileset.json"
    tree = tree_identity(root)
    manifest["stagingId"] = f"staging:{tree['contentDigest']}"
    manifest["tileset"]["contentDigest"] = sha256_file(tileset_path)
    manifest["tileset"]["contentTree"] = tree
    manifest["files"] = file_inventory(root)
    manifest["validation"]["outputBytes"] = tree["bytes"]
    manifest["validation"]["maxOutputBytes"] = tree["bytes"]


def file_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path).removeprefix("sha256:"),
            "content_type": ("model/gltf-binary" if path.suffix == ".glb" else "application/json"),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def write_build_report(root: Path) -> dict[str, object]:
    toolchain = {
        "basisUniversalVersion": "1.16.3",
        "basisUniversalGitCommit": "1531cfaf9ed5232248a0a45736686a849ca3befc",
        "basisUniversalSourceSha256": ("b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac"),
        "basisUniversalBinarySha256": "b" * 64,
        "policySha256": "c" * 64,
    }
    report: dict[str, object] = {
        "gpuTextures": {
            "codec": "etc1s",
            "geometryAndUvBytesPreserved": True,
            "models": [{"path": "0/0.glb", "toolchain": toolchain}],
        }
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["reportPayloadDigest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    (root / "build-report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def staging_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "staging"
    write_glb(root / "0/0.glb", triangles=4)
    tileset = {
        "asset": {"version": "1.0"},
        "geometricError": 10.0,
        "root": {
            "boundingVolume": {"region": [1.0, 0.1, 1.1, 0.2, 2.0, 3.0]},
            "geometricError": 0.0,
            "transform": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            "content": {"uri": "0/0.glb"},
        },
    }
    (root / "tileset.json").write_text(json.dumps(tileset))
    (root / "source-leaf-manifest.json").write_text("{}\n")
    report = write_build_report(root)
    tree = tree_identity(root)
    manifest = {
        "schemaVersion": "mesh-staging-manifest.v2",
        "stagingId": f"staging:{tree['contentDigest']}",
        "candidateId": "mesh:remesh:v55-runtime",
        "source": {
            "odxOutputManifestDigest": "sha256:" + "a" * 64,
            "checkpointId": "checkpoint:odx:fixture",
        },
        "coordinates": {
            "horizontal": "EPSG:32650",
            "vertical": "unspecified",
            "axisOrder": "east,north,up",
            "originOffsetMeters": [0, 0, 0],
        },
        "pipeline": {
            "modelVersion": "v55",
            "taskRevision": "mesh.remesh-3d-tiles.v2",
            "basisUniversal": {
                "version": "1.16.3",
                "gitCommit": "1531cfaf9ed5232248a0a45736686a849ca3befc",
                "sourceSha256": ("b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac"),
                "binarySha256": "b" * 64,
            },
            "texturePolicyDigest": "sha256:" + "c" * 64,
            "validationReport": {
                "relativePath": "build-report.json",
                "contentDigest": sha256_file(root / "build-report.json"),
                "payloadDigest": report["reportPayloadDigest"],
            },
        },
        "tileset": {
            "artifactId": "artifact:staging-tileset:fixture",
            "relativePath": "tileset.json",
            "contentDigest": sha256_file(root / "tileset.json"),
            "contentTree": tree,
        },
        "files": file_inventory(root),
        "validation": {
            "referencedContents": 1,
            "missingContents": 0,
            "outputBytes": tree["bytes"],
            "maxOutputBytes": tree["bytes"],
        },
    }
    return root, manifest


def test_staging_gate_validates_structure_textures_coordinates_and_budgets(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)

    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=manifest["validation"]["outputBytes"],
        min_triangles=4,
        min_textured_glbs=1,
    )

    assert report["passed"] is True
    assert report["glb"]["triangles"] == 4
    assert report["glb"]["textured"] == 1
    assert report["coordinates"]["horizontal"] == "EPSG:32650"
    assert report["filesystem"]["contentReferences"] == 1


def test_staging_contract_rejects_non_string_file_digest(tmp_path: Path) -> None:
    _, manifest = staging_fixture(tmp_path)
    manifest["files"][0]["sha256"] = int("1" * 64)
    with pytest.raises(ValueError, match="bare SHA-256"):
        validate_staging_manifest(manifest, CATALOG)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root, manifest: (root / "0/0.glb").write_bytes(b"bad"), "glb"),
        (lambda root, manifest: manifest["coordinates"].update({"horizontal": "EPSG:4326"}), "CRS"),
        (
            lambda root, manifest: manifest["tileset"].update({"contentDigest": "sha256:" + "b" * 64}),
            "digest",
        ),
        (lambda root, manifest: manifest["validation"].update({"missingContents": 1}), "missing"),
    ],
)
def test_staging_gate_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    root, manifest = staging_fixture(tmp_path)
    mutation(root, manifest)

    with pytest.raises(ValueError, match=message):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_does_not_count_unreferenced_textured_glb(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    write_glb(root / "stray.glb", triangles=100)
    tree = tree_identity(root)
    manifest["stagingId"] = f"staging:{tree['contentDigest']}"
    manifest["tileset"]["contentTree"] = tree
    manifest["validation"]["outputBytes"] = tree["bytes"]
    manifest["validation"]["maxOutputBytes"] = tree["bytes"]

    with pytest.raises(ValueError, match="quality budget"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=5,
            min_textured_glbs=1,
        )


def test_staging_gate_rejects_declared_triangles_without_index_bytes(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    write_glb(root / "0/0.glb", triangles=100, stored_triangles=1)

    with pytest.raises(ValueError, match="accessor exceeds"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=100,
            min_textured_glbs=1,
        )


def test_staging_gate_rejects_texture_signature_outside_image_view(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    write_glb(
        root / "0/0.glb",
        image=b"\x89PNG\r\n\x1a\n",
        image_view_length=4,
        trailing=b"\r\n\x1a\n",
    )

    with pytest.raises(ValueError, match="image bufferView"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_allows_mixed_textured_and_untextured_references(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    write_glb(root / "1/1.glb", textured=False)
    tileset_path = root / "tileset.json"
    tileset = json.loads(tileset_path.read_text())
    tileset["root"]["children"] = [
        {
            "boundingVolume": tileset["root"]["boundingVolume"],
            "geometricError": 0,
            "content": {"uri": "1/1.glb"},
        }
    ]
    tileset_path.write_text(json.dumps(tileset))
    refresh_manifest(root, manifest)
    manifest["validation"]["referencedContents"] = 2

    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=2,
        min_textured_glbs=1,
    )

    assert report["glb"] == {"files": 2, "textured": 1, "textures": 1, "triangles": 5}


def test_staging_gate_rejects_misaligned_chunk_and_buffer_view(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    data = bytearray(path.read_bytes())
    json_length = struct.unpack_from("<I", data, 12)[0]
    struct.pack_into("<I", data, 12, json_length - 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="version or length"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )

    write_glb(path)
    document, binary = glb_document(path)
    document["bufferViews"][0]["byteOffset"] = 2
    write_glb_document(path, document, binary)
    with pytest.raises(ValueError, match="alignment or bounds"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_rejects_missing_position_and_zero_count_offset(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    document, binary = glb_document(path)
    document["meshes"][0]["primitives"][0]["attributes"] = None
    write_glb_document(path, document, binary)
    with pytest.raises(ValueError, match="primitive attributes"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )

    write_glb(path)
    document, binary = glb_document(path)
    document["accessors"].append(
        {"bufferView": 0, "byteOffset": 1000, "count": 0, "componentType": 5125, "type": "SCALAR"}
    )
    document["meshes"][0]["primitives"].append({"mode": 4, "indices": 2, "attributes": {"POSITION": 1}})
    write_glb_document(path, document, binary)
    with pytest.raises(ValueError, match="accessor exceeds"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_uses_bounded_transform_tolerance(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    tileset_path = root / "tileset.json"
    tileset = json.loads(tileset_path.read_text())
    tileset["root"]["transform"][12] = 5e-10
    tileset_path.write_text(json.dumps(tileset))
    refresh_manifest(root, manifest)
    assert validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )["passed"]

    tileset["root"]["transform"][12] = 2e-9
    tileset_path.write_text(json.dumps(tileset))
    refresh_manifest(root, manifest)
    with pytest.raises(ValueError, match="translation"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"].update(
                {"index": True}
            ),
            "texture index",
        ),
        (lambda document: document["textures"][0].update({"source": False}), "texture source"),
        (lambda document: document["images"][0].update({"bufferView": False}), "embedded"),
        (lambda document: document["images"][0].update({"mimeType": []}), "embedded"),
        (lambda document: document["accessors"][0].update({"componentType": []}), "componentType"),
    ],
)
def test_staging_gate_rejects_boolean_and_unhashable_indices(tmp_path: Path, mutation, message: str) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    document, binary = glb_document(path)
    mutation(document)
    write_glb_document(path, document, binary)

    with pytest.raises(ValueError, match=message):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


@pytest.mark.parametrize(
    "document",
    [[], {"asset": []}, {"asset": {"version": "2.0"}, "buffers": [[]]}],
)
def test_staging_gate_normalizes_malformed_glb_json_types(tmp_path: Path, document: object) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    _, binary = glb_document(path)
    write_glb_document(path, document, binary)

    with pytest.raises(ValueError, match="malformed staging data"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"coordinates": []},
        {
            "coordinates": {
                "horizontal": "EPSG:32650",
                "vertical": "unspecified",
                "axisOrder": "east,north,up",
                "originOffsetMeters": [10**1000, 0, 0],
            }
        },
    ],
)
def test_staging_gate_normalizes_malformed_manifest_types(tmp_path: Path, manifest: object) -> None:
    root, _ = staging_fixture(tmp_path)

    with pytest.raises(ValueError, match="malformed staging data"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_decodes_meshopt_before_structural_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    uncompressed = path.read_bytes()
    document, binary = glb_document(path)
    document["extensionsUsed"] = ["EXT_meshopt_compression"]
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)

    with pytest.raises(ValueError, match="Meshopt decoder"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )

    commands = []

    def fake_missing_output(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import core.validation.staging_gate as staging_gate

    monkeypatch.setattr(staging_gate.subprocess, "run", fake_missing_output)
    with pytest.raises(ValueError, match="produced no output"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
            meshopt_decoder="/tools/gltf-transform",
        )

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[3]).write_bytes(uncompressed)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(staging_gate.subprocess, "run", fake_run)
    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
        meshopt_decoder="/tools/gltf-transform",
    )

    assert report["passed"] is True
    assert commands[0][:3] == ["/tools/gltf-transform", "copy", str(path)]


def test_staging_gate_accepts_ext_texture_webp_source(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    write_glb(
        path,
        triangles=4,
        image=b"RIFF\x00\x00\x00\x00WEBP",
        image_mime_type="image/webp",
    )
    document, binary = glb_document(path)
    document["extensionsUsed"] = ["EXT_texture_webp"]
    document["textures"] = [{"extensions": {"EXT_texture_webp": {"source": 0}}}]
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)

    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )

    assert report["glb"]["textured"] == 1

    document["extensionsUsed"] = []
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)
    with pytest.raises(ValueError, match="WebP extension is not declared"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )

    document["extensionsUsed"] = ["EXT_texture_webp"]
    document["textures"] = [{"source": 0, "extensions": {"EXT_texture_webp": None}}]
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)
    with pytest.raises(ValueError, match="WebP texture extension is invalid"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_accepts_khr_texture_basisu_source(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    ktx2 = bytearray(104)
    ktx2[:12] = b"\xabKTX 20\xbb\r\n\x1a\n"
    struct.pack_into("<II", ktx2, 20, 64, 64)
    struct.pack_into("<II", ktx2, 36, 1, 1)
    write_glb(path, triangles=4, image=bytes(ktx2), image_mime_type="image/ktx2")
    document, binary = glb_document(path)
    document["extensionsUsed"] = ["KHR_texture_basisu"]
    document["extensionsRequired"] = ["KHR_texture_basisu"]
    document["textures"] = [{"extensions": {"KHR_texture_basisu": {"source": 0}}}]
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)

    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    assert report["glb"]["textured"] == 1

    document["extensionsUsed"] = []
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)
    with pytest.raises(ValueError, match="BasisU extension is not declared"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_staging_gate_accepts_normalized_quantized_positions(tmp_path: Path) -> None:
    root, manifest = staging_fixture(tmp_path)
    path = root / "0/0.glb"
    document, binary = glb_document(path)
    document["extensionsUsed"] = ["KHR_mesh_quantization"]
    position = document["accessors"][1]
    position["componentType"] = 5122
    position["normalized"] = True
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)

    report = validate_staging_publication(
        root,
        manifest,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )

    assert report["glb"]["triangles"] == 4

    position["normalized"] = False
    write_glb_document(path, document, binary)
    refresh_manifest(root, manifest)
    with pytest.raises(ValueError, match="quantized accessor must be normalized"):
        validate_staging_publication(
            root,
            manifest,
            CATALOG,
            max_output_bytes=10_000_000,
            min_triangles=1,
            min_textured_glbs=1,
        )


def test_publish_staging_is_atomic_content_addressed_and_idempotent(tmp_path: Path) -> None:
    root, staging = staging_fixture(tmp_path)
    validation = validate_staging_publication(
        root,
        staging,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    publications = tmp_path / "publications"
    registry = tmp_path / "publication-registry.sqlite3"

    first = publish_staging(
        root,
        staging,
        validation,
        CATALOG,
        publications_root=publications,
        registry_path=registry,
        remesh_strategy_revision="a" * 40,
        owner_job_id="job-first",
        owner_attempt_id="attempt-first",
    )
    second = publish_staging(
        root,
        staging,
        validation,
        CATALOG,
        publications_root=publications,
        registry_path=registry,
        remesh_strategy_revision="a" * 40,
        owner_job_id="job-second",
        owner_attempt_id="attempt-second",
    )

    assert first == second
    directory = publications / first["atomicDirectory"]["directoryName"]
    assert directory.is_dir()
    assert (directory / "tileset.json").is_file()
    assert not list(publications.glob(".tmp-*"))
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM publication_claims").fetchone()[0] == 2
    assert "tmp" not in json.dumps(first)


def test_publish_staging_concurrent_attempts_share_verified_result(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    root, staging = staging_fixture(tmp_path)
    validation = validate_staging_publication(
        root,
        staging,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    publications = tmp_path / "publications"
    registry = tmp_path / "publication-registry.sqlite3"
    barrier = Barrier(2)

    def publish(attempt: str) -> dict[str, object]:
        barrier.wait()
        return publish_staging(
            root,
            staging,
            validation,
            CATALOG,
            publications_root=publications,
            registry_path=registry,
            remesh_strategy_revision="a" * 40,
            owner_job_id="job-shared",
            owner_attempt_id=attempt,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(publish, ["attempt-first", "attempt-second"])

    assert first == second
    assert len([path for path in publications.iterdir() if not path.name.startswith(".tmp-")]) == 1
    assert not list(publications.glob(".tmp-*"))
    with sqlite3.connect(registry) as connection:
        assert connection.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM publication_claims").fetchone()[0] == 2


def test_publish_staging_rejects_failed_gate_without_visible_output(tmp_path: Path) -> None:
    root, staging = staging_fixture(tmp_path)
    publications = tmp_path / "publications"

    with pytest.raises(ValueError, match="validation report"):
        publish_staging(
            root,
            staging,
            {"schemaVersion": "mesh-staging-validation.v1", "passed": False},
            CATALOG,
            publications_root=publications,
            registry_path=tmp_path / "registry.sqlite3",
            remesh_strategy_revision="a" * 40,
            owner_job_id="job-failed",
            owner_attempt_id="attempt-failed",
        )

    assert not publications.exists()


def test_publish_staging_does_not_follow_new_source_symlink(tmp_path: Path) -> None:
    root, staging = staging_fixture(tmp_path)
    validation = validate_staging_publication(
        root,
        staging,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    (root / "late-link").symlink_to("/etc/passwd")
    publications = tmp_path / "publications"

    with pytest.raises(ValueError, match="symbolic link"):
        publish_staging(
            root,
            staging,
            validation,
            CATALOG,
            publications_root=publications,
            registry_path=tmp_path / "registry.sqlite3",
            remesh_strategy_revision="a" * 40,
            owner_job_id="job-symlink",
            owner_attempt_id="attempt-symlink",
        )

    assert not [path for path in publications.iterdir() if not path.name.startswith(".tmp-")]


def test_publish_staging_rejects_symlinked_final_directory(tmp_path: Path) -> None:
    root, staging = staging_fixture(tmp_path)
    validation = validate_staging_publication(
        root,
        staging,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    publications = tmp_path / "publications"
    publications.mkdir()
    final = publications / validation["contentTree"]["contentDigest"].removeprefix("sha256:")
    final.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="publication path is symbolic"):
        publish_staging(
            root,
            staging,
            validation,
            CATALOG,
            publications_root=publications,
            registry_path=tmp_path / "registry.sqlite3",
            remesh_strategy_revision="a" * 40,
            owner_job_id="job-final-link",
            owner_attempt_id="attempt-final-link",
        )


def test_publish_staging_rejects_symlink_created_during_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, staging = staging_fixture(tmp_path)
    validation = validate_staging_publication(
        root,
        staging,
        CATALOG,
        max_output_bytes=10_000_000,
        min_triangles=1,
        min_textured_glbs=1,
    )
    publications = tmp_path / "publications"

    def race_with_symlink(source: Path, target: Path) -> None:
        del source
        target.symlink_to(root, target_is_directory=True)
        raise OSError(errno.EEXIST, "publication appeared during rename")

    monkeypatch.setattr(os, "rename", race_with_symlink)

    with pytest.raises(ValueError, match="symbolic link"):
        publish_staging(
            root,
            staging,
            validation,
            CATALOG,
            publications_root=publications,
            registry_path=tmp_path / "registry.sqlite3",
            remesh_strategy_revision="a" * 40,
            owner_job_id="job-racing-link",
            owner_attempt_id="attempt-racing-link",
        )
