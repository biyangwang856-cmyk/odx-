"""Fail-closed validation gate for one Mesh staging publication."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.artifacts import sha256_file, tree_identity
from core.contracts.staging import validate_staging_manifest
from core.validation.glb_textures import KTX2_MAGIC, ktx2_info
from core.validation.tilesets import scan_filesystem

IMAGE_MAGIC = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/ktx2": (KTX2_MAGIC,),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),
}


@dataclass(frozen=True)
class AccessorLayout:
    count: int
    component_type: int
    element_bytes: int
    view_offset: int
    byte_offset: int
    stride: int


def _finite_numbers(values: object, count: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{name} must contain {count} numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{name} must contain {count} numbers")
    rendered = [float(value) for value in values]
    if not all(math.isfinite(value) for value in rendered):
        raise ValueError(f"{name} must contain finite numbers")
    return rendered


def _validate_coordinates(
    tileset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_horizontal: str,
) -> dict[str, object]:
    coordinates = manifest["coordinates"]
    if coordinates["horizontal"] != expected_horizontal:
        raise ValueError("staging CRS does not match the expected horizontal CRS")
    root = tileset.get("root")
    if not isinstance(root, dict):
        raise ValueError("tileset root is missing")
    transform = _finite_numbers(root.get("transform"), 16, "tileset root transform")
    if transform[3:16:4] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("tileset root transform must be affine")
    offset = _finite_numbers(coordinates["originOffsetMeters"], 3, "originOffsetMeters")
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        for actual, expected in zip(transform[12:15], offset, strict=True)
    ):
        raise ValueError("tileset root transform translation does not match originOffsetMeters")
    region = _finite_numbers(root.get("boundingVolume", {}).get("region"), 6, "root region")
    west, south, east, north, minimum, maximum = region
    if not (-math.pi <= west < east <= math.pi):
        raise ValueError("root region longitude bounds are invalid")
    if not (-math.pi / 2 <= south < north <= math.pi / 2) or minimum > maximum:
        raise ValueError("root region latitude or height bounds are invalid")
    return {
        "horizontal": coordinates["horizontal"],
        "vertical": coordinates["vertical"],
        "axisOrder": coordinates["axisOrder"],
        "originOffsetMeters": offset,
        "rootTransform": transform,
        "rootRegion": region,
    }


def _inspect_glb(path: Path, meshopt_decoder: str | None = None) -> dict[str, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20 or header[:4] != b"glTF":
            raise ValueError(f"invalid GLB header: {path}")
        version, declared_size = struct.unpack_from("<II", header, 4)
        json_bytes, json_type = struct.unpack_from("<II", header, 12)
        if version != 2 or declared_size != size or json_type != 0x4E4F534A or json_bytes % 4:
            raise ValueError(f"invalid GLB version or length: {path}")
        document = json.loads(stream.read(json_bytes).decode("utf-8"))
        if document.get("asset", {}).get("version") != "2.0":
            raise ValueError(f"GLB asset version is not 2.0: {path}")
        extensions_used = document.get("extensionsUsed", [])
        if not isinstance(extensions_used, list) or not all(
            isinstance(extension, str) for extension in extensions_used
        ):
            raise ValueError(f"GLB extensionsUsed is invalid: {path}")
        if "EXT_meshopt_compression" in extensions_used:
            if meshopt_decoder is None:
                raise ValueError(f"GLB Meshopt decoder is required: {path}")
            return _inspect_decoded_meshopt(path, meshopt_decoder)
        binary_header = stream.read(8)
        if len(binary_header) != 8:
            raise ValueError(f"GLB has no BIN chunk: {path}")
        binary_bytes, binary_type = struct.unpack("<II", binary_header)
        binary_start = stream.tell()
        if binary_type != 0x004E4942 or binary_bytes % 4 or binary_start + binary_bytes != size:
            raise ValueError(f"invalid GLB BIN chunk: {path}")
        buffers = document.get("buffers", [])
        if (
            len(buffers) != 1
            or isinstance(buffers[0].get("byteLength"), bool)
            or not isinstance(buffers[0].get("byteLength"), int)
            or buffers[0]["byteLength"] < 0
            or buffers[0]["byteLength"] > binary_bytes
        ):
            raise ValueError(f"GLB buffer length is invalid: {path}")
        buffer_length = buffers[0]["byteLength"]
        views = document.get("bufferViews", [])
        if not isinstance(views, list):
            raise ValueError(f"GLB bufferViews is invalid: {path}")
        for view in views:
            if not isinstance(view, dict) or view.get("buffer") != 0:
                raise ValueError(f"GLB bufferView references an invalid buffer: {path}")
            offset = view.get("byteOffset", 0)
            length = view.get("byteLength")
            stride = view.get("byteStride")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or offset < 0
                or offset % 4
                or length < 0
                or offset + length > buffer_length
                or (
                    stride is not None
                    and (
                        isinstance(stride, bool)
                        or not isinstance(stride, int)
                        or not 4 <= stride <= 252
                        or stride % 4
                    )
                )
            ):
                raise ValueError(f"GLB bufferView alignment or bounds are invalid: {path}")

        images = document.get("images", [])
        textures = document.get("textures", [])
        materials = document.get("materials", [])
        if not all(isinstance(value, list) for value in (images, textures, materials)):
            raise ValueError(f"GLB texture collections are invalid: {path}")
        textured_materials = 0
        for material in materials:
            if not isinstance(material, dict):
                raise ValueError(f"GLB material is invalid: {path}")
            pbr = material.get("pbrMetallicRoughness", {})
            if not isinstance(pbr, dict):
                raise ValueError(f"GLB PBR material is invalid: {path}")
            texture = pbr.get("baseColorTexture")
            if texture is None:
                continue
            if (
                not isinstance(texture, dict)
                or isinstance(texture.get("index"), bool)
                or not isinstance(texture.get("index"), int)
            ):
                raise ValueError(f"GLB material texture index is invalid: {path}")
            texture_index = texture["index"]
            if not 0 <= texture_index < len(textures):
                raise ValueError(f"GLB material texture index is invalid: {path}")
            texture_entry = textures[texture_index]
            if not isinstance(texture_entry, dict):
                raise ValueError(f"GLB texture is invalid: {path}")
            texture_extensions = texture_entry.get("extensions", {})
            if not isinstance(texture_extensions, dict):
                raise ValueError(f"GLB texture extensions are invalid: {path}")
            webp_present = "EXT_texture_webp" in texture_extensions
            basisu_present = "KHR_texture_basisu" in texture_extensions
            if webp_present and basisu_present:
                raise ValueError(f"GLB texture declares multiple encoded variants: {path}")
            webp = texture_extensions.get("EXT_texture_webp")
            basisu = texture_extensions.get("KHR_texture_basisu")
            if webp_present and "EXT_texture_webp" not in extensions_used:
                raise ValueError(f"GLB WebP extension is not declared: {path}")
            if basisu_present and "KHR_texture_basisu" not in extensions_used:
                raise ValueError(f"GLB BasisU extension is not declared: {path}")
            if webp_present and not isinstance(webp, dict):
                raise ValueError(f"GLB WebP texture extension is invalid: {path}")
            if basisu_present and not isinstance(basisu, dict):
                raise ValueError(f"GLB BasisU texture extension is invalid: {path}")
            variant = basisu if basisu_present else webp if webp_present else None
            image_index = variant.get("source") if variant is not None else texture_entry.get("source")
            if (
                isinstance(image_index, bool)
                or not isinstance(image_index, int)
                or not 0 <= image_index < len(images)
            ):
                raise ValueError(f"GLB texture source is invalid: {path}")
            image = images[image_index]
            if not isinstance(image, dict):
                raise ValueError(f"GLB image is invalid: {path}")
            mime_type = image.get("mimeType")
            view_index = image.get("bufferView")
            if (
                not isinstance(mime_type, str)
                or mime_type not in IMAGE_MAGIC
                or (webp_present and mime_type != "image/webp")
                or (basisu_present and mime_type != "image/ktx2")
                or isinstance(view_index, bool)
                or not isinstance(view_index, int)
            ):
                raise ValueError(f"GLB image must be embedded PNG, JPEG, WebP, or KTX2: {path}")
            if not 0 <= view_index < len(views):
                raise ValueError(f"GLB image bufferView is invalid: {path}")
            view = views[view_index]
            offset = int(view.get("byteOffset", 0))
            length = int(view.get("byteLength", 0))
            prefixes = IMAGE_MAGIC[mime_type]
            required = max(len(prefix) for prefix in prefixes)
            if mime_type == "image/webp":
                required = 12
            if length < required:
                raise ValueError(f"GLB image bufferView escapes BIN chunk: {path}")
            stream.seek(binary_start + offset)
            payload = stream.read(length)
            if len(payload) != length or not any(payload.startswith(prefix) for prefix in prefixes):
                raise ValueError(f"GLB embedded texture magic is invalid: {path}")
            if mime_type == "image/webp" and payload[8:12] != b"WEBP":
                raise ValueError(f"GLB embedded WebP magic is invalid: {path}")
            if mime_type == "image/ktx2":
                info = ktx2_info(payload)
                if (
                    info.get("format") != "ktx2"
                    or info.get("width", 0) <= 0
                    or info.get("height", 0) <= 0
                    or info.get("depth") != 0
                    or info.get("layers") != 0
                    or info.get("faces") != 1
                    or info.get("levels") != 1
                ):
                    raise ValueError(f"GLB embedded KTX2 must be single-level 2D: {path}")
            textured_materials += 1
        triangles = _validated_triangle_count(document, views, stream, binary_start, path)
    if triangles <= 0:
        raise ValueError(f"GLB has no triangles: {path}")
    return {"triangles": triangles, "textures": len(images), "materials": textured_materials}


def _inspect_decoded_meshopt(path: Path, executable: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="mesh-staging-meshopt-") as temporary:
        decoded = Path(temporary) / path.name
        try:
            completed = subprocess.run(
                [executable, "copy", str(path), str(decoded)],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise ValueError(f"GLB Meshopt decoder could not run: {path}") from error
        if completed.returncode != 0:
            detail = (completed.stdout + completed.stderr).strip()[-1000:]
            raise ValueError(f"GLB Meshopt decoding failed: {path}: {detail}")
        if decoded.is_symlink() or not decoded.is_file():
            raise ValueError(f"GLB Meshopt decoder produced no output: {path}")
        try:
            return _inspect_glb(decoded)
        except OSError as error:
            raise ValueError(f"GLB Meshopt decoder output could not be read: {path}") from error


def _nonnegative_int(value: object, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"GLB {name} is invalid: {path}")
    return value


def _validated_accessor(
    document: Mapping[str, Any],
    views: list[dict[str, Any]],
    accessor_index: object,
    *,
    expected_type: str,
    allowed_components: Mapping[int, tuple[int, int]],
    normalized_integer: bool = False,
    path: Path,
) -> AccessorLayout:
    accessors = document.get("accessors", [])
    if (
        isinstance(accessor_index, bool)
        or not isinstance(accessor_index, int)
        or not isinstance(accessors, list)
        or not 0 <= accessor_index < len(accessors)
    ):
        raise ValueError(f"GLB primitive accessor is invalid: {path}")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict) or accessor.get("type") != expected_type:
        raise ValueError(f"GLB accessor type is invalid: {path}")
    component_type = accessor.get("componentType")
    if (
        isinstance(component_type, bool)
        or not isinstance(component_type, int)
        or component_type not in allowed_components
    ):
        raise ValueError(f"GLB accessor componentType is invalid: {path}")
    if normalized_integer and component_type != 5126 and accessor.get("normalized") is not True:
        raise ValueError(f"GLB quantized accessor must be normalized: {path}")
    view_index = accessor.get("bufferView")
    if isinstance(view_index, bool) or not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise ValueError(f"GLB accessor bufferView is invalid: {path}")
    count = _nonnegative_int(accessor.get("count"), "accessor count", path)
    byte_offset = _nonnegative_int(accessor.get("byteOffset", 0), "accessor byteOffset", path)
    element_bytes, alignment = allowed_components[component_type]
    stride = views[view_index].get("byteStride", element_bytes)
    view_length = views[view_index]["byteLength"]
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < element_bytes
        or byte_offset % alignment
        or byte_offset > view_length
        or (count and byte_offset + (count - 1) * stride + element_bytes > view_length)
    ):
        raise ValueError(f"GLB accessor exceeds its bufferView: {path}")
    return AccessorLayout(
        count=count,
        component_type=component_type,
        element_bytes=element_bytes,
        view_offset=views[view_index].get("byteOffset", 0),
        byte_offset=byte_offset,
        stride=stride,
    )


def _maximum_index(stream: Any, binary_start: int, layout: AccessorLayout, path: Path) -> int:
    if layout.count == 0:
        return -1
    formats = {5121: "B", 5123: "H", 5125: "I"}
    format_code = formats[layout.component_type]
    start = binary_start + layout.view_offset + layout.byte_offset
    stream.seek(start)
    if layout.stride == layout.element_bytes:
        raw = stream.read(layout.count * layout.element_bytes)
        if len(raw) != layout.count * layout.element_bytes:
            raise ValueError(f"GLB index accessor is truncated: {path}")
        return max(value[0] for value in struct.iter_unpack(f"<{format_code}", raw))
    maximum = -1
    for index in range(layout.count):
        stream.seek(start + index * layout.stride)
        raw = stream.read(layout.element_bytes)
        if len(raw) != layout.element_bytes:
            raise ValueError(f"GLB index accessor is truncated: {path}")
        maximum = max(maximum, struct.unpack(f"<{format_code}", raw)[0])
    return maximum


def _validated_triangle_count(
    document: Mapping[str, Any],
    views: list[dict[str, Any]],
    stream: Any,
    binary_start: int,
    path: Path,
) -> int:
    meshes = document.get("meshes", [])
    if not isinstance(meshes, list):
        raise ValueError(f"GLB meshes are invalid: {path}")
    triangles = 0
    for mesh in meshes:
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives", []), list):
            raise ValueError(f"GLB mesh primitives are invalid: {path}")
        for primitive in mesh["primitives"]:
            if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                continue
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise ValueError(f"GLB primitive attributes are invalid: {path}")
            extensions_used = document.get("extensionsUsed", [])
            quantized = "KHR_mesh_quantization" in extensions_used
            positions = _validated_accessor(
                document,
                views,
                attributes.get("POSITION"),
                expected_type="VEC3",
                allowed_components=(
                    {
                        5120: (3, 1),
                        5121: (3, 1),
                        5122: (6, 2),
                        5123: (6, 2),
                        5126: (12, 4),
                    }
                    if quantized
                    else {5126: (12, 4)}
                ),
                normalized_integer=quantized,
                path=path,
            )
            if "indices" in primitive:
                indices = _validated_accessor(
                    document,
                    views,
                    primitive["indices"],
                    expected_type="SCALAR",
                    allowed_components={5121: (1, 1), 5123: (2, 2), 5125: (4, 4)},
                    path=path,
                )
                if indices.stride != indices.element_bytes:
                    raise ValueError(f"GLB index accessor cannot be interleaved: {path}")
                if _maximum_index(stream, binary_start, indices, path) >= positions.count:
                    raise ValueError(f"GLB index exceeds POSITION count: {path}")
                count = indices.count
            else:
                count = positions.count
            if count % 3:
                raise ValueError(f"GLB triangle accessor count is not divisible by three: {path}")
            triangles += count // 3
    return triangles


def validate_staging_publication(
    root: Path,
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    max_output_bytes: int | None,
    min_triangles: int,
    min_textured_glbs: int,
    expected_horizontal: str = "EPSG:32650",
    meshopt_decoder: str | None = None,
) -> dict[str, object]:
    """Validate a complete staging directory before any publication mutation."""
    try:
        return _validate_staging_publication(
            root,
            manifest,
            catalog,
            max_output_bytes=max_output_bytes,
            min_triangles=min_triangles,
            min_textured_glbs=min_textured_glbs,
            expected_horizontal=expected_horizontal,
            meshopt_decoder=meshopt_decoder,
        )
    except ValueError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        UnicodeDecodeError,
        struct.error,
    ) as error:
        raise ValueError(f"malformed staging data: {error}") from error


def _validate_staging_publication(
    root: Path,
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    max_output_bytes: int | None,
    min_triangles: int,
    min_textured_glbs: int,
    expected_horizontal: str,
    meshopt_decoder: str | None,
) -> dict[str, object]:
    filesystem, scan = scan_filesystem(root)
    if not filesystem["passed"]:
        raise ValueError(f"staging filesystem gate failed: {filesystem['failures']}")
    glb_paths = sorted({path for path in scan.contents if path.suffix.lower() == ".glb"})
    if not glb_paths:
        raise ValueError("staging gate found no GLB content")
    glb_reports = [_inspect_glb(path, meshopt_decoder) for path in glb_paths]
    triangles = sum(report["triangles"] for report in glb_reports)
    textured = sum(report["materials"] > 0 for report in glb_reports)
    if triangles < min_triangles or textured < min_textured_glbs:
        raise ValueError("staging quality budget failed")

    tileset_path = root / "tileset.json"
    tileset = json.loads(tileset_path.read_text(encoding="utf-8"))
    coordinates = _validate_coordinates(tileset, manifest, expected_horizontal)
    validate_staging_manifest(manifest, catalog)
    if manifest["tileset"]["contentDigest"] != sha256_file(tileset_path):
        raise ValueError("staging tileset digest mismatch")
    identity = tree_identity(root)
    if manifest["tileset"]["contentTree"] != identity:
        raise ValueError("staging content tree digest mismatch")
    output_bytes = identity["bytes"]
    if (
        (max_output_bytes is not None and output_bytes > max_output_bytes)
        or output_bytes != manifest["validation"]["outputBytes"]
        or manifest["validation"]["missingContents"] != 0
        or manifest["validation"]["referencedContents"] != filesystem["contentReferences"]
    ):
        raise ValueError("staging size or missing-content budget failed")
    return {
        "schemaVersion": "mesh-staging-validation.v1",
        "passed": True,
        "candidateId": manifest["candidateId"],
        "stagingId": manifest["stagingId"],
        "filesystem": filesystem,
        "coordinates": coordinates,
        "contentTree": identity,
        "glb": {
            "files": len(glb_paths),
            "textured": textured,
            "textures": sum(report["textures"] for report in glb_reports),
            "triangles": triangles,
        },
        "budgets": {
            "maxOutputBytes": max_output_bytes,
            "minTriangles": min_triangles,
            "minTexturedGlbs": min_textured_glbs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--max-output-bytes", type=int, help="Optional explicit byte ceiling")
    parser.add_argument("--min-triangles", required=True, type=int)
    parser.add_argument("--min-textured-glbs", required=True, type=int)
    parser.add_argument("--expected-horizontal", default="EPSG:32650")
    parser.add_argument(
        "--meshopt-decoder",
        default=os.environ.get("GLTF_TRANSFORM") or shutil.which("gltf-transform"),
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = validate_staging_publication(
        args.root,
        json.loads(args.manifest.read_text()),
        json.loads(args.catalog.read_text()),
        max_output_bytes=args.max_output_bytes,
        min_triangles=args.min_triangles,
        min_textured_glbs=args.min_textured_glbs,
        expected_horizontal=args.expected_horizontal,
        meshopt_decoder=args.meshopt_decoder,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
