#!/usr/bin/env python3
"""Add measured, colored mesh proxies to empty internal detail tiles."""

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from core.geometry.draco_error import (
    SurfaceGeometry,
    accessor_array,
    bidirectional_surface_distances,
    glb_chunks,
    surface_geometry,
    weighted_quantile,
)


def align(data: bytearray, multiple: int = 4) -> None:
    data.extend(b"\0" * ((-len(data)) % multiple))


def link_content(source: str, destination: str) -> str:
    if Path(source).suffix.lower() == ".json":
        return shutil.copy2(source, destination)
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def progress(message: str) -> None:
    """Best-effort progress output; a closed SSH pipe must not abort a build."""
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


@dataclasses.dataclass(frozen=True)
class LeafGeometry:
    positions: np.ndarray
    colors: np.ndarray
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    texture_group = parser.add_mutually_exclusive_group(required=True)
    texture_group.add_argument("--texture-manifest", type=Path)
    texture_group.add_argument("--texture-page-manifest", type=Path)
    parser.add_argument("--ktx-cli", type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--texture-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--texture-sample-size", type=int, default=1024)
    parser.add_argument("--texture-inset", type=float, default=0.05)
    parser.add_argument("--base-ratio", type=float, default=0.01)
    parser.add_argument("--ratio-multiplier", type=float, default=2.0)
    parser.add_argument("--max-ratio", type=float, default=0.2)
    parser.add_argument("--base-error", type=float, default=0.01)
    parser.add_argument("--error-decay", type=float, default=2.0)
    parser.add_argument("--min-error", type=float, default=0.0005)
    parser.add_argument("--error-safety-factor", type=float, default=1.25)
    parser.add_argument("--measurement-quantile", type=float, default=0.99)
    parser.add_argument("--measurement-safety-factor", type=float, default=1.25)
    parser.add_argument("--min-descendant-leaves", type=int, default=2)
    parser.add_argument("--max-p99-error-m", type=float, default=2.0)
    parser.add_argument("--shards", help="Comma-separated shard names for a partial POC")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.input / "tileset.json",
        args.texture_manifest or args.texture_page_manifest,
        args.cli,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if min(args.workers, args.texture_workers, args.texture_sample_size) < 1:
        raise ValueError("worker counts and texture-sample-size must be positive")
    if not 0 <= args.texture_inset < 1:
        raise ValueError("texture-inset must satisfy 0 <= inset < 1")
    if not 0 < args.base_ratio <= args.max_ratio <= 1:
        raise ValueError("ratios must satisfy 0 < base <= max <= 1")
    if args.ratio_multiplier < 1:
        raise ValueError("ratio-multiplier must be at least one")
    if not 0 < args.min_error <= args.base_error <= 1:
        raise ValueError("errors must satisfy 0 < min <= base <= 1")
    if args.error_decay < 1 or args.error_safety_factor < 1:
        raise ValueError("error decay and safety factor must be at least one")
    if not 0 < args.measurement_quantile < 1:
        raise ValueError("measurement-quantile must be between zero and one")
    if args.measurement_safety_factor < 1:
        raise ValueError("measurement-safety-factor must be at least one")
    if args.min_descendant_leaves < 2:
        raise ValueError("min-descendant-leaves must be at least two")
    if args.max_p99_error_m <= 0:
        raise ValueError("max-p99-error-m must be positive")
    if args.texture_page_manifest and not args.ktx_cli:
        raise ValueError("--ktx-cli is required with --texture-page-manifest")
    if args.ktx_cli and not args.ktx_cli.is_file():
        raise FileNotFoundError(args.ktx_cli)


def quaternion_matrix(rotation: list[float]) -> np.ndarray:
    x_value, y_value, z_value, w_value = rotation
    norm = math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value + w_value * w_value)
    if norm == 0:
        raise ValueError("Node quaternion has zero length")
    x_value /= norm
    y_value /= norm
    z_value /= norm
    w_value /= norm
    return np.asarray(
        [
            [
                1 - 2 * (y_value * y_value + z_value * z_value),
                2 * (x_value * y_value - z_value * w_value),
                2 * (x_value * z_value + y_value * w_value),
            ],
            [
                2 * (x_value * y_value + z_value * w_value),
                1 - 2 * (x_value * x_value + z_value * z_value),
                2 * (y_value * z_value - x_value * w_value),
            ],
            [
                2 * (x_value * z_value - y_value * w_value),
                2 * (y_value * z_value + x_value * w_value),
                1 - 2 * (x_value * x_value + y_value * y_value),
            ],
        ],
        dtype=np.float64,
    )


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    rotation = quaternion_matrix(node.get("rotation", [0, 0, 0, 1]))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def mesh_world_matrices(document: dict) -> list[tuple[int, np.ndarray]]:
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if scenes:
        scene_index = document.get("scene", 0)
        roots = scenes[scene_index].get("nodes", [])
    else:
        children = {child for node in nodes for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    result = []

    def visit(index: int, parent_matrix: np.ndarray) -> None:
        node = nodes[index]
        world_matrix = parent_matrix @ node_matrix(node)
        if "mesh" in node:
            result.append((node["mesh"], world_matrix))
        for child in node.get("children", []):
            visit(child, world_matrix)

    for root in roots:
        visit(root, np.eye(4, dtype=np.float64))
    return result


def extract_world_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document, binary = glb_chunks(path)
    position_arrays = []
    index_arrays = []
    position_offset = 0
    for primitive, positions, indices in transformed_primitives(document, binary, path):
        position_arrays.append(positions)
        index_arrays.append(indices + position_offset)
        position_offset += positions.shape[0]
    if not position_arrays or not index_arrays:
        raise ValueError(f"No triangle geometry in {path}")
    return np.concatenate(position_arrays), np.concatenate(index_arrays).astype(np.uint32)


def transformed_primitives(document: dict, binary: bytes, path: Path):
    for mesh_index, world_matrix in mesh_world_matrices(document):
        for primitive in document["meshes"][mesh_index].get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError(f"Only TRIANGLES primitives are supported: {path}")
            accessor_index = primitive.get("attributes", {}).get("POSITION")
            if accessor_index is None:
                continue
            positions = np.asarray(
                accessor_array(document, binary, accessor_index)[:, :3],
                dtype=np.float64,
            )
            positions = positions @ world_matrix[:3, :3].T + world_matrix[:3, 3]
            if "indices" in primitive:
                indices = np.asarray(
                    accessor_array(document, binary, primitive["indices"]),
                    dtype=np.int64,
                )
            else:
                indices = np.arange(positions.shape[0], dtype=np.int64)
            if indices.size % 3:
                raise ValueError(f"Triangle index count is invalid: {path}")
            triangles = indices.reshape((-1, 3))
            triangle_positions = positions[triangles]
            valid = (
                np.linalg.norm(
                    np.cross(
                        triangle_positions[:, 1] - triangle_positions[:, 0],
                        triangle_positions[:, 2] - triangle_positions[:, 0],
                    ),
                    axis=1,
                )
                > 0
            )
            yield primitive, positions, triangles[valid].reshape(-1)


class TextureCache:
    def __init__(
        self,
        manifest_path: Path,
        sample_size: int,
        workers: int,
        required_names: set[str] | None = None,
    ):
        manifest = json.loads(manifest_path.read_text())
        self.sources = {
            Path(item["path"]).name: Path(item["source"])
            for item in manifest["encoded"]
            if required_names is None or Path(item["path"]).name in required_names
        }
        if required_names is not None:
            missing = required_names - self.sources.keys()
            if missing:
                raise KeyError(f"Textures absent from shared manifest: {sorted(missing)[:3]}")
        self.sample_size = sample_size
        self.images: dict[str, np.ndarray] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._load_source, name, source): name
                for name, source in self.sources.items()
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                name, image = future.result()
                self.images[name] = image
                if completed % 100 == 0 or completed == len(futures):
                    progress(f"[{completed}/{len(futures)}] texture colors cached")

    def _load_source(self, name: str, source: Path) -> tuple[str, np.ndarray]:
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            image.thumbnail(
                (self.sample_size, self.sample_size),
                Image.Resampling.LANCZOS,
            )
            decoded = np.asarray(image.convert("RGB")).copy()
        return name, decoded

    def get(self, uri: str) -> np.ndarray:
        name = Path(uri).name
        try:
            return self.images[name]
        except KeyError as exception:
            raise KeyError(f"Texture is absent from the shared manifest: {name}") from exception


class PagedTextureCache:
    def __init__(
        self,
        manifest_path: Path,
        texture_root: Path,
        ktx_cli: Path,
        sample_size: int,
        workers: int,
        required_names: set[str],
    ):
        manifest = json.loads(manifest_path.read_text())
        pages = manifest.get("pages", [])
        self.sources = {
            Path(item["path"]).name: texture_root / item["path"]
            for item in pages
            if Path(item["path"]).name in required_names
        }
        missing = required_names - self.sources.keys()
        if missing:
            raise KeyError(f"Texture pages absent from page manifest: {sorted(missing)[:3]}")
        self.ktx_cli = ktx_cli
        self.sample_size = sample_size
        self.images: dict[str, np.ndarray] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._load_page, name, source): name for name, source in self.sources.items()
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                name, image = future.result()
                self.images[name] = image
                if completed % 100 == 0 or completed == len(futures):
                    progress(f"[{completed}/{len(futures)}] KTX2 page colors cached")

    def _load_page(self, name: str, source: Path) -> tuple[str, np.ndarray]:
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temporary.close()
        decoded = Path(temporary.name)
        try:
            run_command(
                [
                    str(self.ktx_cli),
                    "extract",
                    "--transcode",
                    "rgba8",
                    str(source),
                    str(decoded),
                ]
            )
            with Image.open(decoded) as image:
                image.thumbnail(
                    (self.sample_size, self.sample_size),
                    Image.Resampling.LANCZOS,
                )
                pixels = np.asarray(image.convert("RGB")).copy()
            return name, pixels
        finally:
            decoded.unlink(missing_ok=True)

    def get(self, uri: str) -> np.ndarray:
        name = Path(uri).name
        try:
            return self.images[name]
        except KeyError as exception:
            raise KeyError(f"Texture page is absent from the cache: {name}") from exception


def shard_paths_for_args(root: Path, shards: str | None) -> list[Path]:
    paths = sorted((root / "detail/shards").glob("*/tileset.json"))
    if not shards:
        return paths
    requested = {name.strip() for name in shards.split(",") if name.strip()}
    selected = [path for path in paths if path.parent.name in requested]
    found = {path.parent.name for path in selected}
    if found != requested:
        raise ValueError(f"Unknown shards: {sorted(requested - found)}")
    return selected


def referenced_texture_names(shard_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for shard_path in shard_paths:
        tileset = json.loads(shard_path.read_text())
        stack = [tileset["root"]]
        while stack:
            tile = stack.pop()
            stack.extend(tile.get("children", []))
            content = tile.get("content")
            if not content:
                continue
            uri = content.get("uri") or content.get("url")
            if not uri or not uri.lower().endswith(".glb"):
                continue
            document, _ = glb_chunks(shard_path.parent / uri)
            for image in document.get("images", []):
                image_uri = image.get("uri")
                if image_uri:
                    names.add(Path(image_uri).name)
    return names


def transformed_texcoords(texcoords: np.ndarray, texture_info: dict) -> np.ndarray:
    extension = texture_info.get("extensions", {}).get("KHR_texture_transform", {})
    offset = np.asarray(extension.get("offset", [0, 0]), dtype=np.float64)
    scale = np.asarray(extension.get("scale", [1, 1]), dtype=np.float64)
    rotation = float(extension.get("rotation", 0))
    scaled = np.asarray(texcoords[:, :2], dtype=np.float64) * scale
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    result = np.empty_like(scaled)
    result[:, 0] = cosine * scaled[:, 0] - sine * scaled[:, 1] + offset[0]
    result[:, 1] = sine * scaled[:, 0] + cosine * scaled[:, 1] + offset[1]
    return result


def wrap_coordinates(values: np.ndarray, mode: int) -> np.ndarray:
    if mode == 33071:
        return np.clip(values, 0, 1)
    if mode == 33648:
        wrapped = np.mod(values, 2)
        return np.where(wrapped <= 1, wrapped, 2 - wrapped)
    if mode != 10497:
        raise ValueError(f"Unsupported texture wrap mode: {mode}")
    return np.mod(values, 1)


def sample_texture(
    image: np.ndarray,
    texcoords: np.ndarray,
    wrap_s: int = 10497,
    wrap_t: int = 10497,
) -> np.ndarray:
    u_values = wrap_coordinates(texcoords[:, 0], wrap_s)
    v_values = wrap_coordinates(texcoords[:, 1], wrap_t)
    height, width = image.shape[:2]
    x_values = u_values * (width - 1)
    y_values = v_values * (height - 1)
    x0 = np.floor(x_values).astype(np.int64)
    y0 = np.floor(y_values).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x_weight = (x_values - x0)[:, None]
    y_weight = (y_values - y0)[:, None]
    top = image[y0, x0] * (1 - x_weight) + image[y0, x1] * x_weight
    bottom = image[y1, x0] * (1 - x_weight) + image[y1, x1] * x_weight
    return np.rint(top * (1 - y_weight) + bottom * y_weight).astype(np.uint8)


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    normalized = np.asarray(colors, dtype=np.float64) / 255
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def sample_inset_vertex_colors(
    image: np.ndarray,
    texcoords: np.ndarray,
    indices: np.ndarray,
    inset: float,
    wrap_s: int = 10497,
    wrap_t: int = 10497,
) -> np.ndarray:
    triangles = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    triangle_texcoords = np.asarray(texcoords, dtype=np.float64)[triangles]
    centroids = triangle_texcoords.mean(axis=1, keepdims=True)
    inset_texcoords = triangle_texcoords + inset * (centroids - triangle_texcoords)
    corner_colors = sample_texture(
        image,
        inset_texcoords.reshape((-1, 2)),
        wrap_s,
        wrap_t,
    ).astype(np.float64)
    flat_indices = triangles.reshape(-1)
    vertex_count = len(texcoords)
    counts = np.bincount(flat_indices, minlength=vertex_count)
    sums = np.column_stack(
        [
            np.bincount(flat_indices, weights=corner_colors[:, channel], minlength=vertex_count)
            for channel in range(3)
        ]
    )
    colors = np.empty((vertex_count, 3), dtype=np.uint8)
    used = counts > 0
    colors[used] = np.rint(sums[used] / counts[used, None]).clip(0, 255).astype(np.uint8)
    if not np.all(used):
        colors[~used] = sample_texture(image, texcoords[~used], wrap_s, wrap_t)
    return colors


def primitive_colors(
    document: dict,
    binary: bytes,
    primitive: dict,
    texture_cache: TextureCache,
    vertex_count: int,
    indices: np.ndarray,
    texture_inset: float,
) -> np.ndarray:
    material_index = primitive.get("material")
    if material_index is None:
        return np.full((vertex_count, 3), 255, dtype=np.uint8)
    material = document["materials"][material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3])
    texture_info = pbr.get("baseColorTexture")
    if not texture_info:
        return np.broadcast_to(np.rint(factor * 255), (vertex_count, 3)).astype(np.uint8).copy()
    extension = texture_info.get("extensions", {}).get("KHR_texture_transform", {})
    texcoord_index = extension.get("texCoord", texture_info.get("texCoord", 0))
    semantic = f"TEXCOORD_{texcoord_index}"
    accessor_index = primitive.get("attributes", {}).get(semantic)
    if accessor_index is None:
        raise ValueError(f"Material requires missing attribute {semantic}")
    texcoords = accessor_array(document, binary, accessor_index)
    texcoords = transformed_texcoords(texcoords, texture_info)
    texture = document["textures"][texture_info["index"]]
    image_index = (
        texture.get("extensions", {}).get("KHR_texture_basisu", {}).get("source", texture.get("source"))
    )
    if image_index is None:
        raise ValueError("Base color texture has no source image")
    image = texture_cache.get(document["images"][image_index]["uri"])
    sampler = document.get("samplers", [{}])[texture.get("sampler", 0)]
    colors = sample_inset_vertex_colors(
        image,
        texcoords,
        indices,
        texture_inset,
        sampler.get("wrapS", 10497),
        sampler.get("wrapT", 10497),
    )
    linear_colors = srgb_to_linear(colors)
    return np.rint(linear_colors * factor * 255).clip(0, 255).astype(np.uint8)


def extract_textured_world_geometry(
    path: Path, texture_cache: TextureCache, texture_inset: float
) -> LeafGeometry:
    document, binary = glb_chunks(path)
    positions = []
    colors = []
    indices = []
    position_offset = 0
    for primitive, primitive_positions, primitive_indices in transformed_primitives(document, binary, path):
        positions.append(primitive_positions)
        colors.append(
            primitive_colors(
                document,
                binary,
                primitive,
                texture_cache,
                primitive_positions.shape[0],
                primitive_indices,
                texture_inset,
            )
        )
        indices.append(primitive_indices + position_offset)
        position_offset += primitive_positions.shape[0]
    if not positions:
        raise ValueError(f"No textured triangle geometry in {path}")
    return LeafGeometry(
        positions=np.concatenate(positions),
        colors=np.concatenate(colors),
        indices=np.concatenate(indices).astype(np.uint32),
    )


def load_leaf_geometry(
    path: Path,
    texture_cache: TextureCache,
    texture_inset: float,
) -> LeafGeometry:
    return extract_textured_world_geometry(path, texture_cache, texture_inset)


def write_colored_glb(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray,
    indices: np.ndarray,
) -> float:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = (minimum + maximum) / 2
    local_positions = (positions - center).astype("<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    indices = np.asarray(indices, dtype="<u4")
    binary = bytearray()
    offsets = []
    for payload in (local_positions.tobytes(), colors.tobytes(), indices.tobytes()):
        offsets.append(len(binary))
        binary.extend(payload)
        align(binary)
    document = {
        "asset": {"version": "2.0", "generator": "ODX intermediate mesh HLOD"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": center.astype(float).tolist()}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 1, 1, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                },
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(local_positions),
                "type": "VEC3",
                "min": local_positions.min(axis=0).astype(float).tolist(),
                "max": local_positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "normalized": True,
                "count": len(colors),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": offsets[0],
                "byteLength": local_positions.nbytes,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[1],
                "byteLength": colors.nbytes,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[2],
                "byteLength": indices.nbytes,
                "target": 34963,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    json_data = json.dumps(document, separators=(",", ":")).encode()
    json_data += b" " * ((-len(json_data)) % 4)
    total_length = 12 + 8 + len(json_data) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total_length))
        output.write(struct.pack("<II", len(json_data), 0x4E4F534A))
        output.write(json_data)
        output.write(struct.pack("<II", len(binary), 0x004E4942))
        output.write(binary)
    return float(np.linalg.norm(maximum - minimum) * 0.5)


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def simplifier_error(log: str) -> float:
    values = [float(match.group(1)) for match in re.finditer(r"\berror:\s*([0-9]+(?:\.[0-9]+)?)", log)]
    if not values:
        raise ValueError("glTF-Transform did not report the achieved simplification error")
    return max(values)


def combine_geometry(
    leaf_names: list[str], cache: dict[str, LeafGeometry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    colors = []
    indices = []
    offset = 0
    for name in leaf_names:
        geometry = cache[name]
        positions.append(geometry.positions)
        colors.append(geometry.colors)
        indices.append(geometry.indices.astype(np.uint64) + offset)
        offset += geometry.positions.shape[0]
    return (
        np.concatenate(positions),
        np.concatenate(colors),
        np.concatenate(indices).astype(np.uint32),
    )


def surface_geometry_from_arrays(positions: np.ndarray, indices: np.ndarray) -> SurfaceGeometry:
    positions = np.asarray(positions, dtype=np.float64)
    triangles = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    triangle_positions = positions[triangles]
    twice_area = np.linalg.norm(
        np.cross(
            triangle_positions[:, 1] - triangle_positions[:, 0],
            triangle_positions[:, 2] - triangle_positions[:, 0],
        ),
        axis=1,
    )
    valid = twice_area > 0
    valid_triangles = triangles[valid]
    referenced, remapped = np.unique(valid_triangles.reshape(-1), return_inverse=True)
    weights = np.zeros(len(positions), dtype=np.float64)
    np.add.at(
        weights,
        valid_triangles.reshape(-1),
        np.repeat(twice_area[valid] / 6.0, 3),
    )
    return SurfaceGeometry(
        points=positions[referenced],
        triangle_indices=remapped.reshape((-1, 3)).astype(np.int64),
        point_area_weights=weights[referenced],
        degenerate_triangles=int(np.count_nonzero(~valid)),
        surface_area_m2=float(np.sum(twice_area[valid]) * 0.5),
    )


def measured_proxy_error(
    positions: np.ndarray,
    indices: np.ndarray,
    destination: Path,
    quantile: float,
) -> dict:
    source = surface_geometry_from_arrays(positions, indices)
    candidate = surface_geometry(destination)
    distances, weights = bidirectional_surface_distances(source, candidate)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError(f"Surface measurement has no positive area weight: {destination}")
    return {
        "method": "VTK bidirectional vertex-to-triangle distance",
        "samples": int(distances.size),
        "areaWeightedMeanM": float(np.average(distances, weights=weights)),
        "areaWeightedP95M": weighted_quantile(distances, weights, 0.95),
        "areaWeightedP99M": weighted_quantile(distances, weights, 0.99),
        "areaWeightedQuantile": quantile,
        "areaWeightedQuantileM": weighted_quantile(distances, weights, quantile),
        "maximumM": float(distances.max()),
        "sourceSurfaceAreaM2": source.surface_area_m2,
        "candidateSurfaceAreaM2": candidate.surface_area_m2,
        "candidateToSourceAreaRatio": candidate.surface_area_m2 / source.surface_area_m2,
    }


def process_shard(
    shard_path: Path,
    proxy_dir: Path,
    texture_cache: TextureCache,
    args: argparse.Namespace,
) -> dict:
    shard_name = shard_path.parent.name
    tileset = json.loads(shard_path.read_text())
    leaf_uris = set()

    def collect(tile: dict) -> None:
        content = tile.get("content")
        children = tile.get("children", [])
        if content and not children:
            key = "uri" if "uri" in content else "url"
            uri = content[key]
            if not uri.lower().endswith(".glb"):
                raise ValueError(f"Unexpected detail leaf content: {uri}")
            leaf_uris.add(uri)
        for child in children:
            collect(child)

    collect(tileset["root"])
    cache = {
        uri: load_leaf_geometry(
            shard_path.parent / uri,
            texture_cache,
            args.texture_inset,
        )
        for uri in sorted(leaf_uris)
    }
    reports = []
    proxy_index = 0
    skipped_internal_nodes = 0

    def add_proxy(tile: dict, depth: int) -> tuple[list[str], float]:
        nonlocal proxy_index, skipped_internal_nodes
        children = tile.get("children", [])
        if not children:
            tile["geometricError"] = 0.0
            tile["refine"] = "REPLACE"
            content = tile.get("content")
            if not content:
                return [], 0.0
            key = "uri" if "uri" in content else "url"
            return [content[key]], 0.0
        leaf_names = []
        child_error = 0.0
        for child in children:
            descendant_names, descendant_error = add_proxy(child, depth + 1)
            leaf_names.extend(descendant_names)
            child_error = max(child_error, descendant_error)
        if not leaf_names:
            raise ValueError(f"Internal detail node has no descendant content: {shard_name}")
        if len(leaf_names) < args.min_descendant_leaves:
            skipped_internal_nodes += 1
            tile.pop("content", None)
            tile["geometricError"] = child_error
            tile["refine"] = "REPLACE"
            return leaf_names, child_error
        positions, colors, indices = combine_geometry(leaf_names, cache)
        ratio = min(
            args.max_ratio,
            args.base_ratio * args.ratio_multiplier**depth,
        )
        error_fraction = max(
            args.min_error,
            args.base_error / args.error_decay**depth,
        )
        name = f"mesh-{shard_name}-{proxy_index:03d}.glb"
        proxy_index += 1
        destination = proxy_dir / name
        with tempfile.TemporaryDirectory(prefix=f"mesh-hlod-{shard_name}-") as temp_dir:
            temp_root = Path(temp_dir)
            raw = temp_root / "raw.glb"
            welded = temp_root / "welded.glb"
            radius = write_colored_glb(raw, positions, colors, indices)
            weld_log = run_command(
                [
                    str(args.cli),
                    "weld",
                    str(raw),
                    str(welded),
                ]
            )
            simplify_log = run_command(
                [
                    str(args.cli),
                    "simplify",
                    str(welded),
                    str(destination),
                    "--ratio",
                    str(ratio),
                    "--error",
                    str(error_fraction),
                    "--lock-border",
                    "false",
                    "--verbose",
                ]
            )
        output_geometry = surface_geometry(destination)
        achieved_error_fraction = simplifier_error(simplify_log)
        reported_error_bound = achieved_error_fraction + 0.00005
        simplifier_estimated_error = radius * reported_error_bound * args.error_safety_factor
        measurement = measured_proxy_error(
            positions,
            indices,
            destination,
            args.measurement_quantile,
        )
        candidate_bytes = destination.stat().st_size
        accepted = measurement["areaWeightedP99M"] <= args.max_p99_error_m
        own_error = measurement["areaWeightedQuantileM"] * args.measurement_safety_factor
        geometric_error = max(own_error, child_error) if accepted else child_error
        if accepted:
            tile["content"] = {"uri": f"../../../proxy/{name}"}
        else:
            tile.pop("content", None)
            destination.unlink()
        tile["geometricError"] = geometric_error
        tile["refine"] = "REPLACE"
        reports.append(
            {
                "proxy": name,
                "depth": depth,
                "descendantLeaves": len(leaf_names),
                "sourceTriangles": int(indices.size // 3),
                "outputTriangles": int(output_geometry.triangles - output_geometry.degenerate_triangles),
                "ratioRequested": ratio,
                "ratioAchieved": float(
                    (output_geometry.triangles - output_geometry.degenerate_triangles) / (indices.size // 3)
                ),
                "errorFractionLimit": error_fraction,
                "simplifierErrorFraction": achieved_error_fraction,
                "reportedErrorBoundFraction": reported_error_bound,
                "radiusM": radius,
                "simplifierEstimatedErrorM": simplifier_estimated_error,
                "measuredError": measurement,
                "measurementSafetyFactor": args.measurement_safety_factor,
                "selectionErrorStatistic": "areaWeightedQuantileM",
                "ownMeasuredErrorM": own_error,
                "accepted": accepted,
                "rejectionReason": (
                    None if accepted else f"areaWeightedP99M exceeds {args.max_p99_error_m} m"
                ),
                "geometricErrorM": geometric_error,
                "candidateBytes": candidate_bytes,
                "bytes": candidate_bytes if accepted else 0,
                "weldLog": weld_log.strip().splitlines()[-1:] or [],
                "simplifyLog": [line for line in simplify_log.splitlines() if "error:" in line][-3:],
            }
        )
        return leaf_names, geometric_error

    _, root_error = add_proxy(tileset["root"], 0)
    tileset["geometricError"] = root_error
    shard_path.write_text(json.dumps(tileset, indent=2) + "\n")
    shard_report = {
        "shard": shard_name,
        "leafContents": len(cache),
        "candidates": len(reports),
        "proxies": sum(item["accepted"] for item in reports),
        "rejectedCandidates": sum(not item["accepted"] for item in reports),
        "skippedInternalNodes": skipped_internal_nodes,
        "rootGeometricError": root_error,
        "sourceTriangles": sum(item["sourceTriangles"] for item in reports),
        "outputTriangles": sum(item["outputTriangles"] for item in reports if item["accepted"]),
        "bytes": sum(item["bytes"] for item in reports),
        "nodes": sorted(reports, key=lambda item: item["proxy"]),
    }
    checkpoint = proxy_dir.parent / f"intermediate-mesh-hlod-shard-{shard_name}-report.json"
    checkpoint.write_text(json.dumps(shard_report, indent=2) + "\n")
    return shard_report


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_shards = shard_paths_for_args(args.input, args.shards)
    required_names = referenced_texture_names(input_shards)
    progress(
        f"Caching {len(required_names)} texture assets referenced by {len(input_shards)} selected shards"
    )
    if args.texture_page_manifest:
        texture_cache = PagedTextureCache(
            args.texture_page_manifest,
            args.input / "detail",
            args.ktx_cli,
            args.texture_sample_size,
            args.texture_workers,
            required_names,
        )
        texture_mode = "paged-ktx2"
    else:
        texture_cache = TextureCache(
            args.texture_manifest,
            args.texture_sample_size,
            args.texture_workers,
            required_names,
        )
        texture_mode = "shared-source"
    shutil.copytree(
        args.input,
        args.output,
        copy_function=link_content,
        ignore=shutil.ignore_patterns("intermediate-mesh-hlod-report.json"),
    )
    proxy_dir = args.output / "proxy"
    proxy_dir.mkdir(exist_ok=True)
    shard_paths = shard_paths_for_args(args.output, args.shards)
    requested = {name.strip() for name in args.shards.split(",") if name.strip()} if args.shards else None
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_shard,
                shard_path,
                proxy_dir,
                texture_cache,
                args,
            ): shard_path
            for shard_path in shard_paths
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            report = future.result()
            reports.append(report)
            progress(
                f"[{completed}/{len(futures)}] shard {report['shard']}: "
                f"{report['proxies']} proxies, {report['outputTriangles']} triangles"
            )
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "partial": requested is not None,
        "parameters": {
            "textureMode": texture_mode,
            "referencedTextures": len(required_names),
            "baseRatio": args.base_ratio,
            "ratioMultiplier": args.ratio_multiplier,
            "maxRatio": args.max_ratio,
            "baseError": args.base_error,
            "errorDecay": args.error_decay,
            "minError": args.min_error,
            "errorSafetyFactor": args.error_safety_factor,
            "measurementMethod": "VTK bidirectional vertex-to-triangle distance",
            "measurementQuantile": args.measurement_quantile,
            "measurementSafetyFactor": args.measurement_safety_factor,
            "minDescendantLeaves": args.min_descendant_leaves,
            "maxP99ErrorM": args.max_p99_error_m,
            "textureSampleSize": args.texture_sample_size,
            "textureInset": args.texture_inset,
        },
        "shards": sorted(reports, key=lambda item: item["shard"]),
        "totals": {
            "shards": len(reports),
            "candidates": sum(item["candidates"] for item in reports),
            "proxies": sum(item["proxies"] for item in reports),
            "rejectedCandidates": sum(item["rejectedCandidates"] for item in reports),
            "skippedInternalNodes": sum(item["skippedInternalNodes"] for item in reports),
            "sourceTriangles": sum(item["sourceTriangles"] for item in reports),
            "outputTriangles": sum(item["outputTriangles"] for item in reports),
            "bytes": sum(item["bytes"] for item in reports),
        },
    }
    (args.output / "intermediate-mesh-hlod-report.json").write_text(json.dumps(report, indent=2) + "\n")
    progress(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
