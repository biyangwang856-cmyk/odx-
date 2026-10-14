"""Physical validation of shared ODX orthophoto and textured-mesh outputs."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

from PIL import Image

from core.artifacts import sha256_file
from core.contracts._validation import require_relative_path


def validate_cog(path: Path, expected_crs: str) -> dict[str, object]:
    from osgeo import gdal, osr
    from osgeo_utils.samples.validate_cloud_optimized_geotiff import validate

    gdal.UseExceptions()
    dataset = gdal.Open(str(path))
    if dataset is None or dataset.RasterCount not in (3, 4):
        raise ValueError("orthophoto must be an RGB or RGBA raster")
    _, errors, _ = validate(dataset, full_check=True)
    if errors or dataset.GetMetadata("IMAGE_STRUCTURE").get("LAYOUT") != "COG":
        raise ValueError(f"orthophoto is not a valid COG: {errors}")
    expected = osr.SpatialReference()
    expected.SetFromUserInput(expected_crs)
    actual = dataset.GetSpatialRef()
    if actual is None or not actual.IsSame(expected):
        raise ValueError("orthophoto CRS does not match input")
    transform = dataset.GetGeoTransform()
    if not all(math.isfinite(value) for value in transform) or (
        transform[1] * transform[5] - transform[2] * transform[4] == 0
    ):
        raise ValueError("orthophoto transform is invalid")
    if dataset.RasterXSize <= 0 or dataset.RasterYSize <= 0:
        raise ValueError("orthophoto dimensions must be positive")
    for index in range(1, dataset.RasterCount + 1):
        dataset.GetRasterBand(index).Checksum()
    if dataset.GetRasterBand(1).GetMaskBand().ComputeRasterMinMax()[1] <= 0:
        raise ValueError("orthophoto has no valid pixels")
    return {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "transform": list(transform),
        "crs": expected_crs,
        "overviews": dataset.GetRasterBand(1).GetOverviewCount(),
    }


def read_origin_offset(path: Path, expected_crs: str) -> list[float]:
    from osgeo import osr

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("ODX coords.txt is incomplete")
    header = lines[0].strip()
    match = re.fullmatch(r"WGS84 UTM ([1-9][0-9]?)([NS])", header)
    if match:
        zone = int(match[1])
        if zone > 60:
            raise ValueError("ODX UTM zone is invalid")
        header = f"EPSG:{(32600 if match[2] == 'N' else 32700) + zone}"
    actual, expected = osr.SpatialReference(), osr.SpatialReference()
    if actual.SetFromUserInput(header) != 0 or expected.SetFromUserInput(expected_crs) != 0:
        raise ValueError("ODX coordinate reference is invalid")
    if not actual.IsSame(expected):
        raise ValueError("mesh CRS does not match input")
    offset = [float(value) for value in lines[1].split()]
    if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
        raise ValueError("ODX coordinate offset is invalid")
    return [*offset, 0.0]


def validate_textured_mesh(obj: Path, mtl: Path) -> list[Path]:
    vertices = faces = texture_coordinates = 0
    highest_vertex = highest_texture = 0
    libraries: set[str] = set()
    used_materials: set[str] = set()
    active_material: str | None = None
    with obj.open(encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            if fields[0] in {"v", "vt"}:
                components = 3 if fields[0] == "v" else 2
                if len(fields) < components + 1 or not all(
                    math.isfinite(float(value)) for value in fields[1 : components + 1]
                ):
                    raise ValueError("mesh has invalid coordinates")
                vertices += fields[0] == "v"
                texture_coordinates += fields[0] == "vt"
            elif fields[0] == "mtllib":
                libraries.update(shlex.split(line)[1:])
            elif fields[0] == "usemtl":
                active_material = line.partition(" ")[2].strip()
            elif fields[0] == "f":
                if not active_material:
                    raise ValueError("mesh face has no material")
                used_materials.add(active_material)
                if len(fields) < 4:
                    raise ValueError("mesh face has fewer than three vertices")
                for reference in fields[1:]:
                    indices = reference.split("/")
                    if len(indices) < 2 or not indices[1]:
                        raise ValueError("mesh face has no texture coordinates")
                    vertex, texture = int(indices[0]), int(indices[1])
                    if vertex <= 0 or texture <= 0:
                        raise ValueError("ODX mesh indices must be positive")
                    highest_vertex = max(highest_vertex, vertex)
                    highest_texture = max(highest_texture, texture)
                faces += 1
    if vertices < 3 or faces == 0 or highest_vertex > vertices or highest_texture > texture_coordinates:
        raise ValueError("mesh geometry or indices are incomplete")
    if libraries != {mtl.name} or not used_materials:
        raise ValueError("mesh material library is incomplete")
    materials: set[str] = set()
    textured_materials: set[str] = set()
    current_material: str | None = None
    textures: set[Path] = set()
    with mtl.open(encoding="utf-8") as source:
        for line in source:
            fields = shlex.split(line, comments=True)
            if not fields:
                continue
            if fields[0] == "newmtl" and len(fields) == 2:
                current_material = fields[1]
                materials.add(fields[1])
            elif fields[0] == "map_Kd":
                if len(fields) != 2 or current_material is None:
                    raise ValueError("ODX diffuse texture reference is invalid")
                relative = require_relative_path(fields[1], "mesh texture")
                texture = mtl.parent / relative
                if texture.is_symlink() or not texture.is_file():
                    raise ValueError("mesh texture is missing or a symbolic link")
                with Image.open(texture) as image:
                    image.load()
                textures.add(texture)
                textured_materials.add(current_material)
    if not used_materials <= materials or not used_materials <= textured_materials or not textures:
        raise ValueError("mesh materials or textures are incomplete")
    return sorted(textures)


def output_files(manifest: dict[str, object]) -> list[dict[str, object]]:
    artifacts = manifest["artifacts"]
    mesh = artifacts["mesh"]
    return [artifacts["orthophoto"], mesh["obj"], mesh["mtl"], *mesh["textures"]]


def verify_output_files(root: Path, manifest: dict[str, object]) -> None:
    for artifact in output_files(manifest):
        relative = require_relative_path(artifact["relativePath"], "artifact relativePath")
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("ODX artifact is missing or escapes its bundle")
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["contentDigest"]:
            raise ValueError("ODX artifact bytes do not match the manifest")
