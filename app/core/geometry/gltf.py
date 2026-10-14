#!/usr/bin/env python3
"""Low-level glTF geometry helpers shared by publication quality tools."""

import json
import math
import struct
from pathlib import Path

import numpy as np

COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def glb_chunks(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a GLB: {path}")
    if len(data) < 20:
        raise ValueError(f"Truncated GLB: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk: {path}")
    document = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    offset = 20 + json_length
    if offset == len(data):
        return document, b""
    if offset + 8 > len(data):
        raise ValueError(f"Truncated GLB BIN header: {path}")
    binary_length, binary_type = struct.unpack_from("<II", data, offset)
    if binary_type != 0x004E4942:
        raise ValueError(f"Missing BIN chunk: {path}")
    return document, data[offset + 8 : offset + 8 + binary_length]


def accessor_array(document: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError(f"Sparse accessor {accessor_index} is not supported")
    if "bufferView" not in accessor:
        raise ValueError(f"Accessor {accessor_index} has no bufferView")
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError(f"Accessor {accessor_index} does not use the GLB BIN buffer")
    try:
        dtype = COMPONENT_DTYPES[accessor["componentType"]]
        components = TYPE_COMPONENTS[accessor["type"]]
    except KeyError as error:
        raise ValueError(f"Unsupported accessor {accessor_index}: {error}") from error
    count = int(accessor["count"])
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * components))
    if stride < dtype.itemsize * components:
        raise ValueError(f"Accessor {accessor_index} has an invalid byteStride")
    array = np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()
    return array[:, 0] if accessor["type"] == "SCALAR" else array


def node_local_matrix(node: dict) -> np.ndarray:
    transform_keys = {"translation", "rotation", "scale"}
    if "matrix" in node and transform_keys.intersection(node):
        raise ValueError("A glTF node cannot define both matrix and TRS transforms")
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError("Node matrix must contain 16 finite values")
        matrix = values.reshape((4, 4), order="F")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError("Projective node matrices are not supported")
        return matrix

    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("Node translation must contain three finite values")
    if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
        raise ValueError("Node rotation must contain four finite values")
    if scale.shape != (3,) or not np.all(np.isfinite(scale)):
        raise ValueError("Node scale must contain three finite values")
    norm = float(np.linalg.norm(rotation))
    if norm <= 0.0:
        raise ValueError("Node rotation quaternion cannot be zero")
    x, y, z, w = rotation / norm
    rotation_matrix = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def node_world_matrices(document: dict) -> list[np.ndarray]:
    nodes = document.get("nodes", [])
    parents: list[int | None] = [None] * len(nodes)
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            if not isinstance(child, int) or not 0 <= child < len(nodes):
                raise ValueError(f"Node {parent_index} has an invalid child index")
            if parents[child] is not None:
                raise ValueError(f"Node {child} has multiple parents")
            parents[child] = parent_index

    state = [0] * len(nodes)
    world: list[np.ndarray | None] = [None] * len(nodes)

    def visit(index: int) -> np.ndarray:
        if state[index] == 1:
            raise ValueError("Node graph contains a cycle")
        if state[index] == 2:
            return world[index]  # type: ignore[return-value]
        state[index] = 1
        local = node_local_matrix(nodes[index])
        parent = parents[index]
        result = local if parent is None else visit(parent) @ local
        world[index] = result
        state[index] = 2
        return result

    return [visit(index) for index in range(len(nodes))]


def mesh_instance_matrices(document: dict) -> list[list[np.ndarray]]:
    meshes = document.get("meshes", [])
    instances: list[list[np.ndarray]] = [[] for _ in meshes]
    world = node_world_matrices(document)
    for node_index, node in enumerate(document.get("nodes", [])):
        if "mesh" not in node:
            continue
        mesh_index = node["mesh"]
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            raise ValueError(f"Node {node_index} has an invalid mesh index")
        instances[mesh_index].append(world[node_index])
    identity = np.eye(4, dtype=np.float64)
    return [matrices or [identity] for matrices in instances]


def meshopt_simplification_scale(document: dict) -> float:
    """Return a conservative world-space scale matching meshopt getScale."""
    maximum = 0.0
    instances = mesh_instance_matrices(document)
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        linear_scale = max(
            float(np.linalg.svd(matrix[:3, :3], compute_uv=False)[0]) for matrix in instances[mesh_index]
        )
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None:
                continue
            accessor = document["accessors"][position_index]
            if "min" not in accessor or "max" not in accessor:
                raise ValueError(
                    f"Mesh {mesh_index} primitive {primitive_index} POSITION accessor "
                    "requires min/max bounds for absolute-error simplification"
                )
            lower = np.asarray(accessor["min"], dtype=np.float64)
            upper = np.asarray(accessor["max"], dtype=np.float64)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(upper < lower)
            ):
                raise ValueError(f"Mesh {mesh_index} primitive {primitive_index} has invalid POSITION bounds")
            local_scale = float(np.max(upper - lower))
            maximum = max(maximum, local_scale * linear_scale)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("GLB has no positive finite triangle-mesh simplification scale")
    return maximum


def triangle_count(document: dict) -> int:
    total = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                continue
            if "indices" in primitive:
                count = int(document["accessors"][primitive["indices"]]["count"])
            else:
                position_index = primitive.get("attributes", {}).get("POSITION")
                if position_index is None:
                    continue
                count = int(document["accessors"][position_index]["count"])
            if count % 3:
                raise ValueError("TRIANGLES accessor count is not divisible by three")
            total += count // 3
    return total
