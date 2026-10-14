#!/usr/bin/env python3
"""Measure world-space surface error introduced by simplification and Draco."""

import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.geometry.gltf import accessor_array, glb_chunks, mesh_instance_matrices

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
except ModuleNotFoundError:
    bundled_vtk = Path(__file__).resolve().parents[3] / "third_party/3d-tiles/python-vtk"
    if bundled_vtk.is_dir():
        sys.path.insert(0, str(bundled_vtk))
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


@dataclasses.dataclass(frozen=True)
class SurfaceGeometry:
    points: np.ndarray
    triangle_indices: np.ndarray
    point_area_weights: np.ndarray
    degenerate_triangles: int
    surface_area_m2: float

    @property
    def triangles(self) -> int:
        return int(self.triangle_indices.shape[0] + self.degenerate_triangles)


def surface_geometry_from_document(document: dict, binary: bytes) -> SurfaceGeometry:
    point_arrays = []
    triangle_arrays = []
    weight_arrays = []
    degenerate_count = 0
    surface_area = 0.0
    point_offset = 0
    instances = mesh_instance_matrices(document)
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        for world_matrix in instances[mesh_index]:
            for primitive in mesh.get("primitives", []):
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only TRIANGLES primitives are supported")
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
                    raise ValueError("TRIANGLES index count is not divisible by three")
                triangles = indices.reshape((-1, 3))
                if triangles.size and (
                    int(triangles.min()) < 0 or int(triangles.max()) >= positions.shape[0]
                ):
                    raise ValueError("Triangle index is outside the POSITION accessor")
                triangle_positions = positions[triangles]
                twice_area = np.linalg.norm(
                    np.cross(
                        triangle_positions[:, 1] - triangle_positions[:, 0],
                        triangle_positions[:, 2] - triangle_positions[:, 0],
                    ),
                    axis=1,
                )
                valid = twice_area > 0.0
                degenerate_count += int(np.count_nonzero(~valid))
                surface_area += float(np.sum(twice_area[valid]) * 0.5)
                if np.any(valid):
                    referenced, remapped = np.unique(triangles[valid].reshape(-1), return_inverse=True)
                    point_arrays.append(positions[referenced])
                    triangle_arrays.append(remapped.reshape((-1, 3)).astype(np.int64) + point_offset)
                    triangle_areas = twice_area[valid] * 0.5
                    point_weights = np.zeros(positions.shape[0], dtype=np.float64)
                    np.add.at(
                        point_weights,
                        triangles[valid].reshape(-1),
                        np.repeat(triangle_areas / 3.0, 3),
                    )
                    weight_arrays.append(point_weights[referenced])
                    point_offset += referenced.size
    if not point_arrays:
        raise ValueError("No non-degenerate triangle surfaces")
    return SurfaceGeometry(
        points=np.concatenate(point_arrays),
        triangle_indices=np.concatenate(triangle_arrays),
        point_area_weights=np.concatenate(weight_arrays),
        degenerate_triangles=degenerate_count,
        surface_area_m2=surface_area,
    )


def surface_geometry(path: Path) -> SurfaceGeometry:
    document, binary = glb_chunks(path)
    return surface_geometry_from_document(document, binary)


def vtk_polydata(geometry: SurfaceGeometry):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(geometry.points, deep=True))
    offsets = np.arange(
        0,
        geometry.triangle_indices.size + 1,
        3,
        dtype=np.int64,
    )
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(geometry.triangle_indices.reshape(-1)), deep=True),
    )
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def bidirectional_surface_distances(
    before: SurfaceGeometry, after: SurfaceGeometry
) -> tuple[np.ndarray, np.ndarray]:
    distance = vtk.vtkDistancePolyDataFilter()
    distance.SetInputData(0, vtk_polydata(before))
    distance.SetInputData(1, vtk_polydata(after))
    distance.SignedDistanceOff()
    distance.ComputeSecondDistanceOn()
    distance.Update()
    first = distance.GetOutput(0).GetPointData().GetArray("Distance")
    second = distance.GetOutput(1).GetPointData().GetArray("Distance")
    if first is None or second is None:
        raise RuntimeError("VTK did not produce bidirectional surface distances")
    values = np.concatenate(
        (
            np.asarray(vtk_to_numpy(first), dtype=np.float64),
            np.asarray(vtk_to_numpy(second), dtype=np.float64),
        )
    )
    weights = np.concatenate((before.point_area_weights, after.point_area_weights))
    return values, weights


def compare_one(
    source: Path, compressed: Path, cli: Path, label: str
) -> tuple[str, np.ndarray, np.ndarray, SurfaceGeometry, SurfaceGeometry]:
    with tempfile.NamedTemporaryFile(suffix=".glb") as decoded:
        subprocess.run(
            [str(cli), "copy", str(compressed), decoded.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        before = surface_geometry(source)
        after = surface_geometry(Path(decoded.name))
    distances, weights = bidirectional_surface_distances(before, after)
    return (
        label,
        distances,
        weights,
        before,
        after,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile * cumulative[-1])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--area-weighted-quantile", type=float, default=0.99)
    parser.add_argument("--position-tolerance-meters", type=float, default=0.01)
    args = parser.parse_args()
    if not 0.0 < args.area_weighted_quantile < 1.0:
        raise SystemExit("--area-weighted-quantile must be between zero and one")
    if args.position_tolerance_meters <= 0.0:
        raise SystemExit("--position-tolerance-meters must be positive")

    sources = sorted(args.before.rglob("*.glb"))
    if not sources:
        raise SystemExit(f"No source GLBs below {args.before}")
    pairs = [(source, args.after / source.relative_to(args.before)) for source in sources]
    missing = [str(after) for _, after in pairs if not after.is_file()]
    if missing:
        raise SystemExit(f"Missing compressed GLBs: {missing[:3]}")

    distances = []
    area_weights = []
    worst_tile = None
    worst_distance = -1.0
    source_triangles = 0
    compressed_triangles = 0
    source_degenerate_triangles = 0
    compressed_degenerate_triangles = 0
    source_surface_area_m2 = 0.0
    compressed_surface_area_m2 = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                compare_one,
                source,
                after,
                args.cli,
                str(source.relative_to(args.before)),
            ): source
            for source, after in pairs
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            tile_name, tile_distances, tile_weights, before, after = future.result()
            distances.append(tile_distances)
            area_weights.append(tile_weights)
            source_triangles += before.triangles
            compressed_triangles += after.triangles
            source_degenerate_triangles += before.degenerate_triangles
            compressed_degenerate_triangles += after.degenerate_triangles
            source_surface_area_m2 += before.surface_area_m2
            compressed_surface_area_m2 += after.surface_area_m2
            tile_max = float(tile_distances.max())
            if tile_max > worst_distance:
                worst_distance = tile_max
                worst_tile = tile_name
            if completed % 100 == 0 or completed == len(futures):
                print(f"[{completed}/{len(futures)}] compared", flush=True)

    values = np.concatenate(distances)
    weights = np.concatenate(area_weights)
    total_weight = float(np.sum(weights))
    tolerance_mask = values > args.position_tolerance_meters
    source_nondegenerate_triangles = source_triangles - source_degenerate_triangles
    compressed_nondegenerate_triangles = compressed_triangles - compressed_degenerate_triangles
    report = {
        "tiles": len(pairs),
        "bidirectional_point_to_triangle_samples": int(values.size),
        "source_triangles": source_triangles,
        "compressed_triangles": compressed_triangles,
        "source_degenerate_triangles": source_degenerate_triangles,
        "compressed_degenerate_triangles": compressed_degenerate_triangles,
        "nondegenerate_triangle_delta": (compressed_nondegenerate_triangles - source_nondegenerate_triangles),
        "nondegenerate_triangle_loss_ratio": max(
            0,
            source_nondegenerate_triangles - compressed_nondegenerate_triangles,
        )
        / source_nondegenerate_triangles,
        "source_surface_area_m2": source_surface_area_m2,
        "compressed_surface_area_m2": compressed_surface_area_m2,
        "surface_area_ratio": compressed_surface_area_m2 / source_surface_area_m2,
        "area_weighted_mean_m": float(np.average(values, weights=weights)),
        "area_weighted_p95_m": weighted_quantile(values, weights, 0.95),
        "area_weighted_p99_m": weighted_quantile(values, weights, 0.99),
        "area_weighted_quantile": args.area_weighted_quantile,
        "area_weighted_quantile_m": weighted_quantile(values, weights, args.area_weighted_quantile),
        "position_tolerance_m": args.position_tolerance_meters,
        "sample_fraction_above_position_tolerance": float(np.mean(tolerance_mask)),
        "area_weight_fraction_above_position_tolerance": float(
            np.sum(weights[tolerance_mask]) / total_weight
        ),
        "mean_m": float(values.mean()),
        "p50_m": float(np.quantile(values, 0.50)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "max_m": float(values.max()),
        "fraction_le_0_001m": float(np.mean(values <= 0.001)),
        "fraction_le_0_01m": float(np.mean(values <= 0.01)),
        "fraction_le_0_1m": float(np.mean(values <= 0.1)),
        "samples_gt_0_1m": int(np.sum(values > 0.1)),
        "area_weight_fraction_gt_0_001m": float(np.sum(weights[values > 0.001]) / total_weight),
        "area_weight_fraction_gt_0_01m": float(np.sum(weights[values > 0.01]) / total_weight),
        "area_weight_fraction_gt_0_1m": float(np.sum(weights[values > 0.1]) / total_weight),
        "worst_tile": worst_tile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
