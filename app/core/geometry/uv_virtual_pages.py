#!/usr/bin/env python3
"""Split textured triangles on canonical source-image page boundaries."""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

Vector = tuple[float, ...]


@dataclass(frozen=True)
class ClipVertex:
    source_uv: tuple[float, float]
    attributes: Mapping[str, Vector]


@dataclass(frozen=True, order=True)
class VirtualPage:
    column: int
    row: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def size(self) -> tuple[int, int]:
        return self.right - self.left, self.bottom - self.top


@dataclass(frozen=True)
class PageTriangle:
    page: VirtualPage
    vertices: tuple[ClipVertex, ClipVertex, ClipVertex]

    def local_uvs(self, crop_box: tuple[int, int, int, int] | None = None) -> tuple[tuple[float, float], ...]:
        left, top, right, bottom = crop_box or self.page.box
        width = right - left
        height = bottom - top
        return tuple(
            (
                (vertex.source_uv[0] - left) / width,
                (bottom - vertex.source_uv[1]) / height,
            )
            for vertex in self.vertices
        )


@dataclass(frozen=True)
class PageContentConfig:
    page_size: int
    gutter: int
    resolution_numerator: int = 1
    resolution_denominator: int = 1
    codec: str = "png"
    codec_options: tuple[tuple[str, str], ...] = ()
    color_space: str = "srgb"
    mip_filter: str = "lanczos"
    mip_levels: int = 0

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if self.gutter < 0:
            raise ValueError("gutter must not be negative")
        if self.resolution_numerator < 1 or self.resolution_denominator < 1:
            raise ValueError("resolution ratio must be positive")
        if self.mip_levels < 0:
            raise ValueError("mip_levels must not be negative")


def _lerp(start: Vector, end: Vector, amount: float) -> Vector:
    if len(start) != len(end):
        raise ValueError("Vertex attribute dimensions do not match")
    return tuple(a + (b - a) * amount for a, b in zip(start, end))


def _normalized(vector: Vector) -> Vector:
    length = math.sqrt(sum(value * value for value in vector))
    return vector if length == 0 else tuple(value / length for value in vector)


def interpolate_vertex(start: ClipVertex, end: ClipVertex, amount: float) -> ClipVertex:
    if start.attributes.keys() != end.attributes.keys():
        raise ValueError("Vertex attribute sets do not match")
    attributes = {}
    for name in start.attributes:
        value = _lerp(start.attributes[name], end.attributes[name], amount)
        if name == "NORMAL":
            value = _normalized(value)
        elif name == "TANGENT" and len(value) >= 3:
            value = (*_normalized(value[:3]), *value[3:])
        attributes[name] = value
    return ClipVertex(
        source_uv=(
            start.source_uv[0] + (end.source_uv[0] - start.source_uv[0]) * amount,
            start.source_uv[1] + (end.source_uv[1] - start.source_uv[1]) * amount,
        ),
        attributes=attributes,
    )


def _same_vertex(first: ClipVertex, second: ClipVertex, tolerance: float) -> bool:
    uv_matches = all(abs(first.source_uv[axis] - second.source_uv[axis]) <= tolerance for axis in range(2))
    if not uv_matches or first.attributes.keys() != second.attributes.keys():
        return False
    return all(
        len(first.attributes[name]) == len(second.attributes[name])
        and all(
            math.isclose(first_value, second_value, rel_tol=1e-12, abs_tol=tolerance)
            for first_value, second_value in zip(first.attributes[name], second.attributes[name])
        )
        for name in first.attributes
    )


def _deduplicate_polygon(vertices: Sequence[ClipVertex], tolerance: float) -> list[ClipVertex]:
    result = []
    for vertex in vertices:
        if not result or not _same_vertex(result[-1], vertex, tolerance):
            result.append(vertex)
    if len(result) > 1 and _same_vertex(result[0], result[-1], tolerance):
        result.pop()
    return result


def _clip_edge(
    vertices: Sequence[ClipVertex],
    axis: int,
    boundary: float,
    keep_greater: bool,
    tolerance: float,
) -> list[ClipVertex]:
    if not vertices:
        return []

    def inside(vertex: ClipVertex) -> bool:
        coordinate = vertex.source_uv[axis]
        if keep_greater:
            return coordinate >= boundary - tolerance
        return coordinate <= boundary + tolerance

    output = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if previous_inside != current_inside:
            denominator = current.source_uv[axis] - previous.source_uv[axis]
            if abs(denominator) > tolerance:
                amount = (boundary - previous.source_uv[axis]) / denominator
                output.append(interpolate_vertex(previous, current, amount))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return _deduplicate_polygon(output, tolerance)


def clip_polygon_to_page(
    vertices: Sequence[ClipVertex],
    page: VirtualPage,
    tolerance: float = 1e-9,
) -> list[ClipVertex]:
    clipped = list(vertices)
    for axis, boundary, keep_greater in (
        (0, page.left, True),
        (0, page.right, False),
        (1, page.top, True),
        (1, page.bottom, False),
    ):
        clipped = _clip_edge(clipped, axis, boundary, keep_greater, tolerance)
        if len(clipped) < 3:
            return []
    return clipped


def _triangle_area(vertices: Sequence[ClipVertex]) -> float:
    positions = [vertex.attributes.get("POSITION") for vertex in vertices]
    if any(position is None or len(position) != 3 for position in positions):
        raise ValueError("Every vertex must have a three-component POSITION")
    first, second, third = positions
    ab = tuple(second[axis] - first[axis] for axis in range(3))
    ac = tuple(third[axis] - first[axis] for axis in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return math.sqrt(sum(value * value for value in cross)) / 2


def pages_for_triangle(
    vertices: Sequence[ClipVertex],
    source_size: tuple[int, int],
    page_size: int,
) -> Iterable[VirtualPage]:
    if len(vertices) != 3:
        raise ValueError("Expected one triangle")
    width, height = source_size
    if width < 1 or height < 1 or page_size < 1:
        raise ValueError("Source dimensions and page size must be positive")
    u_values = [vertex.source_uv[0] for vertex in vertices]
    v_values = [vertex.source_uv[1] for vertex in vertices]
    tolerance = 1e-7
    if (
        min(u_values) < -tolerance
        or max(u_values) > width + tolerance
        or min(v_values) < -tolerance
        or max(v_values) > height + tolerance
    ):
        raise ValueError("Repeating or out-of-range UVs cannot use canonical pages")

    def page_range(values: Sequence[float], extent: int) -> tuple[int, int]:
        minimum = max(0.0, min(values))
        maximum = min(float(extent), max(values))
        last_page = (extent - 1) // page_size
        first_index = min(last_page, math.floor(minimum / page_size))
        if math.isclose(minimum, maximum, rel_tol=0, abs_tol=tolerance):
            return first_index, first_index
        last_index = min(
            last_page,
            max(0, math.ceil(maximum / page_size) - 1),
        )
        return first_index, last_index

    minimum_column, maximum_column = page_range(u_values, width)
    minimum_row, maximum_row = page_range(v_values, height)
    for row in range(minimum_row, maximum_row + 1):
        for column in range(minimum_column, maximum_column + 1):
            left = column * page_size
            top = row * page_size
            yield VirtualPage(
                column=column,
                row=row,
                left=left,
                top=top,
                right=min(width, left + page_size),
                bottom=min(height, top + page_size),
            )


def split_triangle_by_pages(
    vertices: Sequence[ClipVertex],
    source_size: tuple[int, int],
    page_size: int,
    area_tolerance: float = 1e-12,
) -> list[PageTriangle]:
    fragments = []
    for page in pages_for_triangle(vertices, source_size, page_size):
        polygon = clip_polygon_to_page(vertices, page)
        for index in range(1, len(polygon) - 1):
            triangle = (polygon[0], polygon[index], polygon[index + 1])
            if _triangle_area(triangle) > area_tolerance:
                fragments.append(PageTriangle(page=page, vertices=triangle))
    return fragments


def page_content_key(
    source_digest: str,
    page: VirtualPage,
    source_size: tuple[int, int],
    config: PageContentConfig,
    pixel_digest: str | None = None,
) -> str:
    record = {
        "schema": "uv-virtual-page-v1",
        "sourceDigest": source_digest.lower(),
        "sourceSize": list(source_size),
        "page": {
            "column": page.column,
            "row": page.row,
            "box": list(page.box),
        },
        "config": {
            "pageSize": config.page_size,
            "gutter": config.gutter,
            "resolution": [
                config.resolution_numerator,
                config.resolution_denominator,
            ],
            "codec": config.codec,
            "codecOptions": list(sorted(config.codec_options)),
            "colorSpace": config.color_space,
            "mipFilter": config.mip_filter,
            "mipLevels": config.mip_levels,
        },
        "pixelDigest": pixel_digest.lower() if pixel_digest else None,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def page_resource_path(content_key: str, suffix: str = ".ktx2") -> str:
    if len(content_key) != 64 or any(character not in "0123456789abcdef" for character in content_key):
        raise ValueError("content_key must be a lowercase SHA-256 digest")
    return f"textures/{content_key[:2]}/{content_key}{suffix}"
