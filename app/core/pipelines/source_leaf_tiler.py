#!/usr/bin/env python3
"""Export textured OBJ source leaves with VTK without merging material UVs."""

import argparse
import hashlib
import json
import math
import os
import shlex
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_TEXTURE_PIXEL_LIMIT = 700_000_000


@dataclass
class MaterialMesh:
    points: list[tuple[float, float, float]] = field(default_factory=list)
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float] | None] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    vertices: dict[tuple, int] = field(default_factory=dict)
    source_page_box: tuple[int, int, int, int] | None = None
    source_texture_size: tuple[int, int] | None = None
    source_page_size: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an OBJ as UV-preserving, material-separated source-leaf 3D Tiles."
    )
    parser.add_argument("--input-obj", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crs", required=True)
    parser.add_argument("--offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--proj-data", type=Path)
    parser.add_argument("--material", action="append", dest="materials")
    parser.add_argument("--features-per-tile", type=int, default=256)
    parser.add_argument("--crop-textures-dir", type=Path)
    parser.add_argument("--crop-padding", type=int, default=2)
    parser.add_argument("--texture-block-size", type=int, default=1)
    parser.add_argument("--texture-workers", type=int, default=4)
    parser.add_argument("--max-texture-size", type=int, default=0)
    parser.add_argument("--uv-page-size", type=int, default=0)
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Reflect OBJ Y into north-positive projected coordinates and preserve face orientation.",
    )
    parser.add_argument(
        "--transform",
        nargs=16,
        type=float,
        help="Column-major 4x4 transform applied to OBJ points before export.",
    )
    return parser.parse_args()


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def resolve_index(index: int, count: int) -> int:
    resolved = index - 1 if index > 0 else count + index
    if not 0 <= resolved < count:
        raise IndexError(f"OBJ index {index} is outside 1..{count}")
    return resolved


def parse_reference(
    reference: str, position_count: int, texcoord_count: int, normal_count: int
) -> tuple[int, int, int | None]:
    parts = reference.split("/")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Textured source leaf has a face without UVs: {reference}")
    position = resolve_index(int(parts[0]), position_count)
    texcoord = resolve_index(int(parts[1]), texcoord_count)
    normal = None
    if len(parts) > 2 and parts[2]:
        normal = resolve_index(int(parts[2]), normal_count)
    return position, texcoord, normal


def obj_material_library(input_obj: Path) -> Path:
    with input_obj.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("mtllib "):
                name = shlex.split(line)[1]
                return input_obj.parent / name
    raise ValueError(f"No mtllib entry in {input_obj}")


def material_textures(input_obj: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    active = ""
    with obj_material_library(input_obj).open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("newmtl "):
                active = line.split(maxsplit=1)[1].strip()
            elif line.lower().startswith("map_kd ") and active:
                mapping[active] = shlex.split(line)[-1]
    return mapping


def load_obj(
    input_obj: Path,
    selected_materials: set[str] | None,
    flip_y: bool = False,
    transform: list[float] | None = None,
) -> tuple[dict[str, MaterialMesh], tuple[float, float, float, float, float, float]]:
    if flip_y and transform:
        raise ValueError("--flip-y and --transform are mutually exclusive")

    def transform_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        if flip_y:
            return point[0], -point[1], point[2]
        if not transform:
            return point
        x, y, z = point
        return (
            transform[0] * x + transform[4] * y + transform[8] * z + transform[12],
            transform[1] * x + transform[5] * y + transform[9] * z + transform[13],
            transform[2] * x + transform[6] * y + transform[10] * z + transform[14],
        )

    def transform_normal(normal: tuple[float, float, float]) -> tuple[float, float, float]:
        if flip_y:
            return normal[0], -normal[1], normal[2]
        if not transform:
            return normal
        x, y, z = normal
        value = (
            transform[0] * x + transform[4] * y + transform[8] * z,
            transform[1] * x + transform[5] * y + transform[9] * z,
            transform[2] * x + transform[6] * y + transform[10] * z,
        )
        length = math.sqrt(sum(component * component for component in value))
        return tuple(component / max(length, 1e-12) for component in value)

    determinant = None
    if transform:
        a, b, c = transform[0], transform[4], transform[8]
        d, e, f = transform[1], transform[5], transform[9]
        g, h, i = transform[2], transform[6], transform[10]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    reverse_winding = flip_y or (determinant is not None and determinant < 0)
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    meshes: dict[str, MaterialMesh] = defaultdict(MaterialMesh)
    active_material = ""
    bounds = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]

    with input_obj.open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            if line.startswith("v "):
                values = line.split()
                point = tuple(float(value) for value in values[1:4])
                positions.append(transform_point(point))
            elif line.startswith("vt "):
                values = line.split()
                texcoords.append((float(values[1]), float(values[2])))
            elif line.startswith("vn "):
                values = line.split()
                normal = tuple(float(value) for value in values[1:4])
                normals.append(transform_normal(normal))
            elif line.startswith("usemtl "):
                active_material = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                if not active_material:
                    raise ValueError(f"Face at line {line_number} has no material")
                if selected_materials is not None and active_material not in selected_materials:
                    continue
                mesh = meshes[active_material]
                face: list[int] = []
                for token in line.split()[1:]:
                    reference = parse_reference(token, len(positions), len(texcoords), len(normals))
                    local_index = mesh.vertices.get(reference)
                    if local_index is None:
                        local_index = len(mesh.points)
                        mesh.vertices[reference] = local_index
                        position_index, texcoord_index, normal_index = reference
                        point = positions[position_index]
                        texcoord = texcoords[texcoord_index]
                        normal = normals[normal_index] if normal_index is not None else None
                        mesh.points.append(point)
                        mesh.texcoords.append(texcoord)
                        mesh.normals.append(normal)
                        bounds[0] = min(bounds[0], point[0])
                        bounds[1] = max(bounds[1], point[0])
                        bounds[2] = min(bounds[2], point[1])
                        bounds[3] = max(bounds[3], point[1])
                        bounds[4] = min(bounds[4], point[2])
                        bounds[5] = max(bounds[5], point[2])
                    face.append(local_index)
                for index in range(1, len(face) - 1):
                    triangle = (face[0], face[index], face[index + 1])
                    mesh.triangles.append(
                        (triangle[0], triangle[2], triangle[1]) if reverse_winding else triangle
                    )

    if not meshes:
        raise ValueError("No OBJ faces matched the requested materials")
    if selected_materials:
        missing = selected_materials.difference(meshes)
        if missing:
            raise ValueError(f"Requested materials are not present: {sorted(missing)}")
    return dict(meshes), tuple(bounds)


def triangle_metrics(meshes: dict[str, MaterialMesh]) -> tuple[str, dict[str, float]]:
    uv_records: list[bytes] = []
    area_sum = 0.0
    edge_length_sum = 0.0
    for mesh in meshes.values():
        for triangle in mesh.triangles:
            uv_vertices = []
            triangle_points = []
            for point_id in triangle:
                point = mesh.points[point_id]
                texcoord = mesh.texcoords[point_id]
                uv = (float32(texcoord[0]), float32(1.0 - float32(texcoord[1])))
                uv_vertices.append(struct.pack("<2f", *uv))
                triangle_points.append(point)
            edges = []
            for start, end in ((0, 1), (1, 2), (2, 0)):
                edges.append(
                    math.sqrt(
                        sum(
                            (triangle_points[start][axis] - triangle_points[end][axis]) ** 2
                            for axis in range(3)
                        )
                    )
                )
            semiperimeter = sum(edges) / 2.0
            area_sum += math.sqrt(
                max(
                    0.0,
                    semiperimeter
                    * (semiperimeter - edges[0])
                    * (semiperimeter - edges[1])
                    * (semiperimeter - edges[2]),
                )
            )
            edge_length_sum += sum(edges)
            uv_records.append(b"".join(sorted(uv_vertices)))
    uv_hash = hashlib.sha256(b"".join(sorted(uv_records))).hexdigest()
    return uv_hash, {"triangleArea": area_sum, "edgeLength": edge_length_sum}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_source_texture(path: Path):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = SOURCE_TEXTURE_PIXEL_LIMIT
    image = Image.open(path)
    if image.width * image.height > SOURCE_TEXTURE_PIXEL_LIMIT:
        image.close()
        raise ValueError(f"Source texture exceeds pixel limit: {path}")
    return image


def split_meshes_by_uv_page(
    meshes: dict[str, MaterialMesh],
    input_obj: Path,
    mapping: dict[str, str],
    page_size: int,
) -> tuple[dict[str, MaterialMesh], dict[str, str], dict[str, str]]:
    if page_size == 0:
        return meshes, mapping, {material: material for material in meshes}
    from core.geometry.uv_virtual_pages import ClipVertex, split_triangle_by_pages

    split_meshes = {}
    split_mapping = {}
    source_materials = {}
    for material, mesh in meshes.items():
        source_path = input_obj.parent / mapping[material]
        with open_source_texture(source_path) as image:
            width, height = image.size
        groups: dict[tuple[int, int], MaterialMesh] = {}
        for triangle in mesh.triangles:
            vertices = []
            triangle_has_normals = all(mesh.normals[index] is not None for index in triangle)
            for source_index in triangle:
                normal = mesh.normals[source_index]
                attributes = {"POSITION": mesh.points[source_index]}
                if triangle_has_normals:
                    attributes["NORMAL"] = normal
                u, v = mesh.texcoords[source_index]
                vertices.append(
                    ClipVertex(
                        source_uv=(u * width, (1.0 - v) * height),
                        attributes=attributes,
                    )
                )
            for page_triangle in split_triangle_by_pages(vertices, (width, height), page_size):
                page = page_triangle.page
                fragment = groups.setdefault(
                    (page.column, page.row),
                    MaterialMesh(
                        source_page_box=page.box,
                        source_texture_size=(width, height),
                        source_page_size=page_size,
                    ),
                )
                remapped = []
                for vertex in page_triangle.vertices:
                    point = vertex.attributes["POSITION"]
                    texcoord = (
                        vertex.source_uv[0] / width,
                        1.0 - vertex.source_uv[1] / height,
                    )
                    normal = vertex.attributes.get("NORMAL")
                    key = (
                        tuple(float32(value) for value in point)
                        + tuple(float32(value) for value in texcoord)
                        + (tuple(float32(value) for value in normal) if normal is not None else ())
                    )
                    target_index = fragment.vertices.get(key)
                    if target_index is None:
                        target_index = len(fragment.points)
                        fragment.vertices[key] = target_index
                        fragment.points.append(point)
                        fragment.texcoords.append(texcoord)
                        fragment.normals.append(normal)
                    remapped.append(target_index)
                fragment.triangles.append(tuple(remapped))

        for page_index, (page, fragment) in enumerate(sorted(groups.items())):
            name = material if len(groups) == 1 else f"{material}__uv_{page[0]}_{page[1]}"
            if name in split_meshes:
                name = f"{name}_{page_index}"
            split_meshes[name] = fragment
            split_mapping[name] = mapping[material]
            source_materials[name] = material
    return split_meshes, split_mapping, source_materials


def crop_texture(job: dict) -> dict:
    from PIL import Image

    with open_source_texture(job["source"]) as image:
        cropped = image.crop(job["cropBox"])
        maximum = job["maxTextureSize"]
        if maximum and max(cropped.size) > maximum:
            cropped.thumbnail((maximum, maximum), resample=Image.Resampling.LANCZOS)
        cropped.save(job["output"], format="PNG")
        output_size = list(cropped.size)
    return {
        **job,
        "sourceSha256": sha256(job["source"]),
        "sha256": sha256(job["output"]),
        "outputSize": output_size,
    }


def prepare_textures(
    meshes: dict[str, MaterialMesh],
    input_obj: Path,
    mapping: dict[str, str],
    crop_dir: Path | None,
    padding: int,
    block_size: int,
    workers: int,
    max_texture_size: int,
) -> tuple[Path, dict[str, str], dict[str, dict]]:
    source_paths = {material: input_obj.parent / mapping[material] for material in meshes}
    if crop_dir is None:
        metadata = {
            material: {
                "sourceUri": mapping[material],
                "sourceSha256": sha256(source_paths[material]),
                "sha256": sha256(source_paths[material]),
            }
            for material in sorted(meshes)
        }
        return input_obj.parent, mapping, metadata

    if crop_dir.exists() and any(crop_dir.iterdir()):
        raise FileExistsError(f"Texture crop directory is not empty: {crop_dir}")
    crop_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    prepared_mapping = {}
    for index, material in enumerate(sorted(meshes)):
        mesh = meshes[material]
        source = source_paths[material]
        with open_source_texture(source) as image:
            width, height = image.size
        u_values = [texcoord[0] for texcoord in mesh.texcoords]
        v_values = [texcoord[1] for texcoord in mesh.texcoords]
        uv_bounds = (min(u_values), max(u_values), min(v_values), max(v_values))
        if uv_bounds[0] < 0 or uv_bounds[1] > 1 or uv_bounds[2] < 0 or uv_bounds[3] > 1:
            raise ValueError(f"Cannot crop repeating UVs for {material}: {uv_bounds}")
        if mesh.source_page_box is not None:
            if mesh.source_texture_size != (width, height):
                raise ValueError(f"Source texture size changed for {material}")
            page_left, page_top, page_right, page_bottom = mesh.source_page_box
            left = max(0, page_left - padding)
            right = min(width, page_right + padding)
            top = max(0, page_top - padding)
            bottom = min(height, page_bottom + padding)
        else:
            left = max(0, math.floor(uv_bounds[0] * width) - padding)
            right = min(width, math.ceil(uv_bounds[1] * width) + padding)
            top = max(0, math.floor((1.0 - uv_bounds[3]) * height) - padding)
            bottom = min(height, math.ceil((1.0 - uv_bounds[2]) * height) + padding)
        extra_width = (-(right - left)) % block_size
        grow_right = min(extra_width, width - right)
        right += grow_right
        left -= extra_width - grow_right
        extra_height = (-(bottom - top)) % block_size
        grow_bottom = min(extra_height, height - bottom)
        bottom += grow_bottom
        top -= extra_height - grow_bottom
        if right <= left or bottom <= top:
            raise ValueError(f"Empty texture crop for {material}: {(left, top, right, bottom)}")
        if left < 0 or top < 0 or (right - left) % block_size or (bottom - top) % block_size:
            raise ValueError(f"Source texture cannot provide a {block_size}x{block_size} crop for {material}")
        crop_width = right - left
        crop_height = bottom - top
        mesh.texcoords = [
            (
                (u * width - left) / crop_width,
                (v * height - height + bottom) / crop_height,
            )
            for u, v in mesh.texcoords
        ]
        output_name = f"texture-{index:04d}.png"
        prepared_mapping[material] = output_name
        jobs.append(
            {
                "material": material,
                "source": source,
                "output": crop_dir / output_name,
                "sourceUri": mapping[material],
                "outputUri": output_name,
                "sourceSize": [width, height],
                "cropBox": [left, top, right, bottom],
                "cropSize": [crop_width, crop_height],
                "maxTextureSize": max_texture_size,
                "uvBounds": list(uv_bounds),
                **(
                    {
                        "virtualPageBox": list(mesh.source_page_box),
                        "virtualPageSize": mesh.source_page_size,
                        "virtualPageGutter": padding,
                    }
                    if mesh.source_page_box is not None
                    else {}
                ),
            }
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(crop_texture, jobs))
    metadata = {
        result["material"]: {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in result.items()
            if key not in ("material", "source", "output")
        }
        for result in results
    }
    return crop_dir, prepared_mapping, metadata


def set_texture_field(vtk, polydata, texture_uri: str) -> None:
    values = vtk.vtkStringArray()
    values.SetName("texture_uri")
    values.InsertNextValue(texture_uri)
    polydata.GetFieldData().AddArray(values)


def vtk_polydata(vtk, mesh: MaterialMesh, texture_uri: str):
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for point in mesh.points:
        points.InsertNextPoint(*point)

    polygons = vtk.vtkCellArray()
    for triangle in mesh.triangles:
        cell = vtk.vtkTriangle()
        for index, point_id in enumerate(triangle):
            cell.GetPointIds().SetId(index, point_id)
        polygons.InsertNextCell(cell)

    texcoords = vtk.vtkFloatArray()
    texcoords.SetName("TCoords")
    texcoords.SetNumberOfComponents(2)
    for texcoord in mesh.texcoords:
        texcoords.InsertNextTuple2(*texcoord)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)
    polydata.GetPointData().SetTCoords(texcoords)
    if all(normal is not None for normal in mesh.normals):
        normal_array = vtk.vtkFloatArray()
        normal_array.SetName("Normals")
        normal_array.SetNumberOfComponents(3)
        for normal in mesh.normals:
            normal_array.InsertNextTuple3(*normal)
        polydata.GetPointData().SetNormals(normal_array)
    set_texture_field(vtk, polydata, texture_uri)
    return polydata


def check_tileset(output_dir: Path) -> list[str]:
    tileset_path = output_dir / "tileset.json"
    tileset = json.loads(tileset_path.read_text())
    uris: list[str] = []

    def visit(tile: dict) -> None:
        volume = tile.get("boundingVolume", {})
        coordinates = next((volume[key] for key in ("region", "box", "sphere") if key in volume), [])
        if not coordinates or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in coordinates
        ):
            raise RuntimeError("VTK emitted an invalid geographic bounding volume")
        content = tile.get("content")
        if content:
            uris.append(content.get("uri", content.get("url")))
        for child in tile.get("children", []):
            visit(child)

    visit(tileset["root"])
    if not uris:
        raise RuntimeError("VTK tileset has no renderable content")
    for uri in uris:
        if not uri or not (output_dir / uri).is_file():
            raise FileNotFoundError(output_dir / str(uri))
    return uris


def main() -> None:
    args = parse_args()
    if args.features_per_tile < 1:
        raise ValueError("features-per-tile must be positive")
    if (
        args.crop_padding < 0
        or args.texture_block_size < 1
        or args.texture_workers < 1
        or args.max_texture_size < 0
        or args.uv_page_size < 0
    ):
        raise ValueError(
            "crop-padding, max-texture-size, and uv-page-size must be non-negative; "
            "texture-block-size and workers must be positive"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    proj_data = args.proj_data or next(
        (
            Path(os.environ[name])
            for name in ("VTK_PROJ_LIB", "PROJ_DATA", "PROJ_LIB")
            if os.environ.get(name)
        ),
        None,
    )
    if proj_data is None:
        raise RuntimeError("Set PROJ_DATA or pass --proj-data so VTK can georeference the tileset")
    if not (proj_data / "proj.db").is_file():
        raise FileNotFoundError(proj_data / "proj.db")
    for name in ("PROJ_DATA", "PROJ_LIB", "VTK_PROJ_LIB"):
        os.environ[name] = str(proj_data)

    selected_materials = set(args.materials) if args.materials else None
    meshes, bounds = load_obj(args.input_obj, selected_materials, args.flip_y, args.transform)
    source_material_count = len(meshes)
    source_triangle_count = sum(len(mesh.triangles) for mesh in meshes.values())
    source_uv_hash, source_geometry_metrics = triangle_metrics(meshes)
    texture_mapping = material_textures(args.input_obj)
    missing_textures = sorted(set(meshes).difference(texture_mapping))
    if missing_textures:
        raise ValueError(f"Materials have no map_Kd texture: {missing_textures}")
    source_texture_paths = {
        material: args.input_obj.parent / texture_mapping[material] for material in meshes
    }
    for path in source_texture_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    meshes, texture_mapping, source_materials = split_meshes_by_uv_page(
        meshes,
        args.input_obj,
        texture_mapping,
        args.uv_page_size,
    )
    texture_base_dir, prepared_mapping, texture_metadata = prepare_textures(
        meshes,
        args.input_obj,
        texture_mapping,
        args.crop_textures_dir,
        args.crop_padding,
        args.texture_block_size,
        args.texture_workers,
        args.max_texture_size,
    )
    prepared_texture_paths = {material: texture_base_dir / prepared_mapping[material] for material in meshes}

    import vtk

    root = vtk.vtkMultiBlockDataSet()
    for block_index, material in enumerate(sorted(meshes)):
        building = vtk.vtkMultiBlockDataSet()
        building.SetBlock(0, vtk_polydata(vtk, meshes[material], prepared_mapping[material]))
        root.SetBlock(block_index, building)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkCesium3DTilesWriter()
    writer.SetInputDataObject(root)
    writer.SetInputType(writer.Buildings)
    writer.SetDirectoryName(str(args.output_dir))
    writer.SetTextureBaseDirectory(str(texture_base_dir))
    writer.SetOffset(*args.offset)
    writer.SetCRS(args.crs)
    writer.SetNumberOfFeaturesPerTile(args.features_per_tile)
    writer.SetSaveTiles(True)
    writer.SetSaveTextures(True)
    writer.SetContentGLTF(True)
    writer.SetContentGLTFSaveGLB(True)
    writer.SetMergeTilePolyData(False)
    writer.Write()
    if writer.GetErrorCode():
        raise RuntimeError(f"VTK writer failed with error code {writer.GetErrorCode()}")

    content_uris = check_tileset(args.output_dir)
    uv_hash, geometry_metrics = triangle_metrics(meshes)
    manifest = {
        "inputObj": str(args.input_obj),
        "crs": args.crs,
        "offset": list(args.offset),
        "flipY": args.flip_y,
        "transform": args.transform,
        "bounds": list(bounds),
        "sourceMaterialCount": source_material_count,
        "sourceTriangleCount": source_triangle_count,
        "sourceUvHash": source_uv_hash,
        "sourceGeometryMetrics": source_geometry_metrics,
        "materialCount": len(meshes),
        "uvPageSize": args.uv_page_size,
        "triangleCount": sum(len(mesh.triangles) for mesh in meshes.values()),
        "pointCount": sum(len(mesh.points) for mesh in meshes.values()),
        "uvHash": uv_hash,
        "geometryMetrics": geometry_metrics,
        "geometryAreaRelativeError": abs(
            geometry_metrics["triangleArea"] - source_geometry_metrics["triangleArea"]
        )
        / max(source_geometry_metrics["triangleArea"], 1e-12),
        "textures": [
            {
                "material": material,
                "sourceMaterial": source_materials[material],
                "uri": prepared_mapping[material],
                "bytes": prepared_texture_paths[material].stat().st_size,
                **texture_metadata[material],
            }
            for material in sorted(meshes)
        ],
        "contentUris": content_uris,
        "contentBytes": sum((args.output_dir / uri).stat().st_size for uri in content_uris),
    }
    manifest_path = args.output_dir / "source-leaf-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
