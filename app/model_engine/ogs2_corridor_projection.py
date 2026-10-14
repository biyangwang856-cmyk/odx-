"""Read-only, hash-bound evidence for coherent orthophoto road corridors.

This module projects real undistorted photographs onto the retained 2.5D
surface. It never adjusts a camera, changes the surface, or writes a mosaic.
The output is evidence, not permission to replace any orthophoto pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage

from model_engine.ogs2 import (
    Config,
    Detector,
    cameras_from_checkpoint,
    read_offset,
    sha256,
    surface_points,
    visibility_mask,
)
from model_engine.ogs2_deghost import capture_time


SCHEMA = "ogs2.corridor-projection/v1"


@dataclass(frozen=True)
class ProjectionConfig:
    max_sources: int = 16
    min_estimated_coverage: float = 0.35
    sample_cells: int = 32
    max_roi_pixels: int = 12_000_000
    projection_rows: int = 192
    visibility_tolerance_m: float = 0.35
    visibility_halo_m: float = 40.1
    detector_tile_size: int = 1024
    detector_halo: int = 128
    projected_detection_pass: bool = False
    cpu_threads: int = 8


def canonical_digest(value: object) -> str:
    """Bind JSON contents, independently of whitespace and dictionary order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(canonical_digest({"shape": list(value.shape), "dtype": value.dtype.str}).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def validate_window(window, width: int, height: int, maximum: int) -> Window:
    values = list(window.flatten()) if isinstance(window, Window) else list(window)
    if len(values) != 4 or not all(np.isfinite(v) and int(v) == v for v in values):
        raise ValueError("Evidence window must have four finite integer pixel values")
    col, row, w, h = map(int, values)
    if col < 0 or row < 0 or w < 2 or h < 2 or col + w > width or row + h > height:
        raise ValueError("Evidence window lies outside the reference grid")
    if w * h > maximum:
        raise ValueError("Evidence window exceeds the configured bounded-memory pixel limit")
    return Window(col, row, w, h)


def bilinear_support(mask: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Require all four source pixels, including zero-weight integer neighbours.

    A photograph's last row/column cannot supply four distinct neighbours.
    NaNs, padding, and a single invalid contributing band reject the sample.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or coordinates.shape[0] != 2:
        raise ValueError("Expected a 2D source-valid mask and 2x... source coordinates")
    x, y = coordinates
    finite = np.isfinite(x) & np.isfinite(y)
    inside = finite & (x >= 0) & (y >= 0) & (x < mask.shape[1] - 1) & (y < mask.shape[0] - 1)
    xi = np.floor(np.where(inside, x, 0)).astype(np.int64)
    yi = np.floor(np.where(inside, y, 0)).astype(np.int64)
    # Degenerate dimensions have no four-pixel support, without indexing +1.
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return np.zeros(x.shape, bool)
    return inside & mask[yi, xi] & mask[yi, xi + 1] & mask[yi + 1, xi] & mask[yi + 1, xi + 1]


def _sample_indices(mask: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """One representative from each occupied spatial cell, not one ROI centre."""
    ys, xs = [], []
    height, width = mask.shape
    for row in np.linspace(0, height, min(cells, height) + 1, dtype=int)[:-1]:
        row_end = min(height, row + math.ceil(height / min(cells, height)))
        for col in np.linspace(0, width, min(cells, width) + 1, dtype=int)[:-1]:
            col_end = min(width, col + math.ceil(width / min(cells, width)))
            yy, xx = np.where(mask[row:row_end, col:col_end])
            if len(xx):
                target = np.array([np.median(xx), np.median(yy)])
                nearest = int(np.argmin((xx - target[0]) ** 2 + (yy - target[1]) ** 2))
                ys.append(row + int(yy[nearest]))
                xs.append(col + int(xx[nearest]))
    if not xs:
        raise ValueError("Corridor has no sampled pixels")
    return np.asarray(ys, np.int32), np.asarray(xs, np.int32)


def rank_corridor_sources(factory, data: dict, points: np.ndarray, corridor: np.ndarray, offset, cells=32):
    """Examine every shot using distributed whole-corridor support estimates.

    These estimates omit opacity and visibility; the full cached valid mask is
    the authority. Camera distance only breaks equal-coverage ties.
    """
    ys, xs = _sample_indices(corridor & np.isfinite(points[2]), cells)
    sample = points[:, ys, xs]
    records = []
    for name in sorted(data["shots"]):
        camera = factory.get(name)
        shot = data["shots"][name]
        coordinates = camera.world_to_pixel(sample)
        rotation = cv2.Rodrigues(np.asarray(shot["rotation"], np.float64))[0]
        z_camera = (rotation @ (sample - np.asarray(offset)[:, None]))[2] + shot["translation"][2]
        iw, ih = camera.im_size
        covered = np.isfinite(coordinates).all(axis=0) & (z_camera > 0)
        covered &= (coordinates[0] >= 0) & (coordinates[0] < iw - 1)
        covered &= (coordinates[1] >= 0) & (coordinates[1] < ih - 1)
        fraction = float(covered.mean())
        if not covered.any():
            continue
        height = np.asarray(camera.pos)[2] - sample[2, covered]
        distance = np.hypot(sample[0, covered] - camera.pos[0], sample[1, covered] - camera.pos[1])
        ratio = distance / np.maximum(height, 1.0)
        stamp, domain = capture_time(name, shot)
        records.append({
            "sourceImage": name,
            "estimatedCorridorCoverage": fraction,
            "sampledSpatialCells": int(len(xs)),
            "coveredSpatialCells": int(covered.sum()),
            "medianNadirDistanceRatio": float(np.median(ratio)),
            "captureTime": stamp,
            "captureTimeDomain": domain,
        })
    return sorted(records, key=lambda r: (-r["estimatedCorridorCoverage"], r["medianNadirDistanceRatio"], r["sourceImage"]))


def load_surface_crop(surface_path: Path, grid_transform: Affine, window: Window, halo_m: float):
    """Read the unchanged height field only within ROI plus the visibility ray halo."""
    left, bottom, right, top = rio.windows.bounds(window, grid_transform)
    with rio.open(surface_path) as src:
        if src.count != 1 or not src.crs or not src.crs.is_projected or src.transform.b or src.transform.d:
            raise ValueError("Require a north-up, projected, single-band surface")
        # A coarse surface can have step >40m; ray_visible also probes 2*step.
        actual_halo = max(halo_m, 2 * abs(src.transform.a), 2 * abs(src.transform.e))
        raw = rio.windows.from_bounds(left - actual_halo, bottom - actual_halo, right + actual_halo, top + actual_halo, src.transform)
        col0, row0 = max(0, math.floor(raw.col_off) - 2), max(0, math.floor(raw.row_off) - 2)
        col1 = min(src.width, math.ceil(raw.col_off + raw.width) + 2)
        row1 = min(src.height, math.ceil(raw.row_off + raw.height) + 2)
        if col1 <= col0 or row1 <= row0:
            raise ValueError("Reference ROI does not intersect the retained surface")
        crop_window = Window(col0, row0, col1 - col0, row1 - row0)
        surface = src.read(1, window=crop_window, masked=True).filled(np.nan).astype(np.float32)
        return surface, src.window_transform(crop_window), src.crs, list(crop_window.flatten())


def _project_stripe(source_path, camera, shot, offset, points, surface, surface_transform, tolerance):
    h, w = points.shape[1:]
    coordinates = camera.world_to_pixel(points.reshape(3, -1)).reshape(2, h, w).astype(np.float32)
    rotation = cv2.Rodrigues(np.asarray(shot["rotation"], np.float64))[0]
    camera_z = (rotation @ (points.reshape(3, -1) - np.asarray(offset)[:, None]))[2]
    camera_z = (camera_z + shot["translation"][2]).reshape(h, w)
    iw, ih = camera.im_size
    inside = np.isfinite(coordinates).all(axis=0) & np.isfinite(points).all(axis=0) & (camera_z > 0)
    inside &= (coordinates[0] >= 0) & (coordinates[0] < iw - 1)
    inside &= (coordinates[1] >= 0) & (coordinates[1] < ih - 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    valid = np.zeros((h, w), bool)
    cost = np.full((h, w), 1e6, np.float32)
    if not inside.any():
        return rgb, valid, cost
    x0 = max(0, int(np.floor(coordinates[0][inside].min())))
    y0 = max(0, int(np.floor(coordinates[1][inside].min())))
    x1 = min(iw, int(np.floor(coordinates[0][inside].max())) + 2)
    y1 = min(ih, int(np.floor(coordinates[1][inside].max())) + 2)
    with rio.open(source_path) as src:
        if (src.width, src.height) != (iw, ih) or src.count < 3 or any(t != "uint8" for t in src.dtypes[:3]):
            raise ValueError("Source image does not match uint8 RGB calibrated camera")
        read_window = Window(x0, y0, x1 - x0, y1 - y0)
        image = src.read([1, 2, 3], window=read_window).transpose(1, 2, 0)
        source_valid = np.all(src.read_masks([1, 2, 3], window=read_window) > 0, axis=0)
    maps = coordinates - np.asarray([x0, y0], np.float32)[:, None, None]
    support = bilinear_support(source_valid, maps)
    safe_maps = np.nan_to_num(maps, nan=-1e6, posinf=-1e6, neginf=-1e6)
    rgb = cv2.remap(image, safe_maps[0], safe_maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    visible = visibility_mask(points, camera.pos, surface, surface_transform, tolerance)
    valid = inside & support & visible
    # Gradient computation has a one-row halo in the caller, avoiding stripe seams.
    du_y, du_x = np.gradient(coordinates[0])
    dv_y, dv_x = np.gradient(coordinates[1])
    determinant = np.abs(du_x * dv_y - du_y * dv_x)
    condition = (du_x ** 2 + du_y ** 2 + dv_x ** 2 + dv_y ** 2) / np.maximum(determinant, 1e-4)
    nadir = np.hypot(points[0] - camera.pos[0], points[1] - camera.pos[1]) / np.maximum(camera.pos[2] - points[2], 1)
    calculated = nadir ** 2 + 0.1 * np.maximum(condition - 2, 0)
    valid &= np.isfinite(calculated)
    cost[valid] = calculated[valid]
    rgb[~valid] = 0
    return rgb, valid, cost


def project_source(project, name, camera, shot, offset, points, surface, surface_transform, config):
    """Project one source in row stripes; returned arrays have the exact ROI grid."""
    h, w = points.shape[1:]
    image = np.zeros((h, w, 3), np.uint8)
    valid = np.zeros((h, w), bool)
    cost = np.full((h, w), 1e6, np.float32)
    for row in range(0, h, config.projection_rows):
        last = min(h, row + config.projection_rows)
        start_halo, end_halo = max(0, row - 1), min(h, last + 1)
        part = _project_stripe(
            project / "opensfm/undistorted/images" / name, camera, shot, offset,
            points[:, start_halo:end_halo], surface, surface_transform, config.visibility_tolerance_m,
        )
        read = np.s_[row - start_halo:last - start_halo]
        image[row:last], valid[row:last], cost[row:last] = (value[read] for value in part)
    return image, valid, cost


def native_projection_agreement(camera, shot, camera_parameters, offset, points, *, tolerance_px=1e-5):
    """Compare Orthority with native OpenSfM undistorted perspective equations.

    OpenSfM denormalization uses (width-1)/2, (height-1)/2 pixel centres.
    The W/2 convention is also recorded to expose its constant half-pixel
    difference. This function never changes either camera or its projection.
    """
    if camera_parameters.get("projection_type") != "perspective" or any(camera_parameters.get(k, 0) != 0 for k in ("k1", "k2")):
        raise ValueError("Native comparison requires an undistorted perspective camera")
    xyz = np.asarray(points, np.float64).reshape(3, -1)
    rotation = cv2.Rodrigues(np.asarray(shot["rotation"], np.float64))[0]
    local = rotation @ (xyz - np.asarray(offset)[:, None]) + np.asarray(shot["translation"])[:, None]
    valid = np.isfinite(local).all(axis=0) & (local[2] > 0)
    if valid.sum() < 3:
        raise ValueError("At least three finite in-front 3D points are required")
    iw, ih = camera.im_size
    focal = float(camera_parameters["focal"]) * max(iw, ih)
    native = local[:2, valid] / local[2, valid] * focal + np.asarray([(iw - 1) / 2, (ih - 1) / 2])[:, None]
    projected = camera.world_to_pixel(xyz[:, valid])
    delta = projected - native
    error = np.hypot(delta[0], delta[1])
    constant = np.median(delta, axis=1)
    residual = delta - constant[:, None]
    return {"testedPoints": int(valid.sum()), "focalPixels": focal,
            "nativePixelCentreConvention": "((width-1)/2, (height-1)/2)",
            "orthorityMinusNativeMedianXY": constant.tolist(),
            "maxNativeDifferencePixels": float(error.max()),
            "maxDifferenceAfterMedianPixels": float(np.hypot(residual[0], residual[1]).max()),
            "orthorityMinusWidthHalfConventionMedianXY": (constant - 0.5).tolist(),
            "accepted": bool(error.max() <= tolerance_px), "tolerancePixels": tolerance_px,
            "interpretation": "Equation agreement is not proof of accurate surface or vehicle shape"}


def detect_projected_bodies(image: np.ndarray, valid: np.ndarray, detector, config: ProjectionConfig):
    """Inspect each real-source projection, retaining explicit crop-edge uncertainty.

    No detections is not a certificate of vehicle-free road. A detector cannot
    certify that this photograph's vehicle was geometrically projected intact.
    """
    h, w = valid.shape
    if detector is None:
        return [], "not-run"
    records = []
    for row in range(0, h, config.detector_tile_size):
        for col in range(0, w, config.detector_tile_size):
            y0, x0 = max(0, row - config.detector_halo), max(0, col - config.detector_halo)
            y1 = min(h, row + config.detector_tile_size + config.detector_halo)
            x1 = min(w, col + config.detector_tile_size + config.detector_halo)
            if not valid[y0:y1, x0:x1].any():
                continue
            for observation in detector.masks(image[y0:y1, x0:x1]):
                local = np.asarray(observation["mask"], bool)
                if local.shape != (y1 - y0, x1 - x0) or not local.any():
                    continue
                ys, xs = np.where(local)
                # Keep an object only in its centre's core tile, except a complete
                # halo observation that may upgrade an incomplete core observation.
                cx, cy = float(np.mean(xs)) + x0, float(np.mean(ys)) + y0
                in_core = col <= cx < min(w, col + config.detector_tile_size) and row <= cy < min(h, row + config.detector_tile_size)
                if not in_core and not observation.get("complete", False):
                    continue
                mask = np.zeros((h, w), bool)
                mask[y0:y1, x0:x1] = local
                margin = min(int(xs.min()) + x0, int(ys.min()) + y0, w - 1 - int(xs.max()) - x0, h - 1 - int(ys.max()) - y0)
                support = ndimage.binary_dilation(mask, iterations=1)
                complete = bool(observation.get("complete", False) and margin >= 5 and valid[support].all())
                records.append({
                    "mask": mask,
                    "complete": complete,
                    "confidence": float(observation.get("confidence", 0)),
                    "samConfidence": float(observation.get("samConfidence", 0)),
                    "box": [float(v) + (x0 if i % 2 == 0 else y0) for i, v in enumerate(observation["box"])],
                    "detectorWindow": [x0, y0, x1 - x0, y1 - y0],
                    "roiMarginPixels": margin,
                    "bodyValidFraction": float(valid[mask].mean()),
                })
    # Suppress overlapping halo observations; a complete observation is preferred
    # over an incomplete crop. Never union partial masks and call that complete.
    ordered = sorted(records, key=lambda r: (not r["complete"], -r["mask"].sum(), -r["confidence"]))
    retained = []
    for record in ordered:
        duplicate = False
        for old in retained:
            overlap = int((record["mask"] & old["mask"]).sum())
            minimum = min(int(record["mask"].sum()), int(old["mask"].sum()))
            if overlap / max(minimum, 1) >= 0.75:
                duplicate = True
                break
        if not duplicate:
            retained.append(record)
    return retained, "executed" if retained else "executed-no-accepted-objects"


def detect_source_bodies(project, name, camera, points, projected_valid, detector, config):
    """Detect vehicles in original-photo context, then project those masks.

    This avoids declaring a warped or broken projected vehicle to be empty
    road simply because its distorted appearance defeated the detector.
    Original-photo completeness and projected support completeness remain
    distinct fields; neither certifies vehicle geometry on the 2.5D surface.
    """
    if detector is None:
        return [], "not-run", None
    h, w = projected_valid.shape
    coordinates = camera.world_to_pixel(points.reshape(3, -1)).reshape(2, h, w).astype(np.float32)
    iw, ih = camera.im_size
    inside = np.isfinite(coordinates).all(axis=0) & np.isfinite(points).all(axis=0)
    inside &= (coordinates[0] >= 0) & (coordinates[0] < iw - 1)
    inside &= (coordinates[1] >= 0) & (coordinates[1] < ih - 1)
    if not inside.any():
        return [], "executed-no-source-support", None
    padding = max(64, config.detector_halo)
    x0 = max(0, int(np.floor(coordinates[0][inside].min())) - padding)
    y0 = max(0, int(np.floor(coordinates[1][inside].min())) - padding)
    x1 = min(iw, int(np.ceil(coordinates[0][inside].max())) + padding + 1)
    y1 = min(ih, int(np.ceil(coordinates[1][inside].max())) + padding + 1)
    with rio.open(project / "opensfm/undistorted/images" / name) as src:
        read_window = Window(x0, y0, x1 - x0, y1 - y0)
        raw = src.read([1, 2, 3], window=read_window).transpose(1, 2, 0)
        raw_valid = np.all(src.read_masks([1, 2, 3], window=read_window) > 0, axis=0)
    detections, state = detect_projected_bodies(raw, raw_valid, detector, config)
    maps = coordinates - np.asarray([x0, y0], np.float32)[:, None, None]
    safe_maps = np.nan_to_num(maps, nan=-1e6, posinf=-1e6, neginf=-1e6)
    support = bilinear_support(raw_valid, maps) & inside
    projected = []
    source_union = np.zeros(raw_valid.shape, bool)
    for observation in detections:
        source_union |= observation["mask"]
        mask = cv2.remap(observation["mask"].astype(np.uint8), safe_maps[0], safe_maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
        mask &= support
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        margin = min(int(xs.min()), int(ys.min()), w - 1 - int(xs.max()), h - 1 - int(ys.max()))
        fully_supported = bool(projected_valid[ndimage.binary_dilation(mask, iterations=1)].all())
        # Detect whether the original object is truncated by the target ROI.
        # Full source area is recorded, but a surface mapping can magnify its
        # target footprint, so counts cannot certify full target coverage.
        projected.append({
            "mask": mask, "complete": bool(observation["complete"] and margin >= 5 and fully_supported),
            "sourcePhotoComplete": observation["complete"], "projectedSupportComplete": fully_supported,
            "detectionOrigin": "original-photo-context", "confidence": observation["confidence"],
            "samConfidence": observation["samConfidence"], "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "sourceBox": [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(observation["box"])],
            "sourceCropWindow": [x0, y0, x1 - x0, y1 - y0],
            "roiMarginPixels": margin, "bodyValidFraction": float(projected_valid[mask].mean()),
        })
    context = {"source_crop_image": raw, "source_crop_valid": raw_valid,
               "source_crop_body": source_union, "source_crop_window": np.asarray([x0, y0, x1 - x0, y1 - y0], np.int32)}
    return projected, state, context


def verify_hash_binding(binding: dict) -> None:
    for path, expected in binding.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError("Evidence input changed: " + path)


def build_corridor_evidence(
    project: Path, grid_path: Path, surface_path: Path, output_dir: Path,
    window, corridor_mask: np.ndarray, *, detector=None, detector_identity: dict | None = None,
    config: ProjectionConfig = ProjectionConfig(), source_names: list[str] | None = None,
) -> Path:
    """Atomically publish a bounded ROI and real-source projection evidence.

    ``window`` is [column,row,width,height] in ``grid_path``; ``corridor_mask`` is
    its boolean shape. ``source_names`` may explicitly request candidates but
    cannot avoid coverage/opacity/visibility checks. All other candidates are
    ranked from every shot, over the full corridor. An existing output refuses
    overwrite; failed outputs retain status/evidence and are not usable caches.
    """
    started = time.monotonic()
    project, grid_path, surface_path, output_dir = (Path(p).resolve() for p in (project, grid_path, surface_path, output_dir))
    if output_dir.is_relative_to(project) or project.is_relative_to(output_dir):
        raise ValueError("Evidence output must be separate from the preserved checkpoint")
    if any(output_dir == p or p.is_relative_to(output_dir) for p in (grid_path, surface_path)):
        raise ValueError("Evidence output overlaps retained raster inputs")
    if config.max_sources < 1 or config.sample_cells < 1 or config.projection_rows < 2 or config.detector_tile_size < 32 or config.detector_halo < 5:
        raise ValueError("Invalid bounded projection configuration")
    if detector is not None and not detector_identity:
        raise ValueError("Detection evidence requires a caller-supplied model/config identity")
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "status.json", {"schema": SCHEMA, "state": "preparing", "config": asdict(config)})
    cv2.setNumThreads(config.cpu_threads)
    cv2.ocl.setUseOpenCL(False)
    try:
        critical = [
            grid_path, surface_path, project / "opensfm/undistorted/reconstruction.json",
            project / "odm_georeferencing/odm_georeferencing_model_geo.txt",
            project / "odm_meshing/odm_25dmesh.ply",
        ]
        if surface_path.with_suffix(".json").is_file():
            critical.append(surface_path.with_suffix(".json"))
        binding = {str(p): sha256(p) for p in critical}
        with rio.open(grid_path) as ref:
            if not ref.crs or not ref.crs.is_projected or ref.transform.b or ref.transform.d:
                raise ValueError("Require a north-up projected orthophoto grid")
            win = validate_window(window, ref.width, ref.height, config.max_roi_pixels)
            shape = (int(win.height), int(win.width))
            corridor = np.asarray(corridor_mask, bool)
            if corridor.shape != shape or not corridor.any():
                raise ValueError("Corridor mask does not match a nonempty evidence window")
            old = ref.read(window=win)
            if old.shape[0] < 3 or ref.dtypes[0] != "uint8":
                raise ValueError("Expected a uint8 RGB reference orthophoto")
            alpha = old[3] if old.shape[0] >= 4 else ref.dataset_mask(window=win)
            rgba = np.concatenate([old[:3].transpose(1, 2, 0), alpha[..., None]], axis=2)
            grid = {"crs": ref.crs.to_string(), "transform": list(ref.window_transform(win))[:6],
                    "width": shape[1], "height": shape[0], "referenceWindow": list(win.flatten()),
                    "referenceTransform": list(ref.transform)[:6], "referenceWidth": ref.width,
                    "referenceHeight": ref.height, "pixelAreaSquareMapUnits": abs(ref.transform.a * ref.transform.e)}
            surface, st, surface_crs, crop_window = load_surface_crop(surface_path, ref.transform, win, config.visibility_halo_m)
            if surface_crs != ref.crs:
                raise ValueError("Surface and orthophoto use different CRS")
            points = surface_points(surface, st, win, ref.transform)
            factory, data = cameras_from_checkpoint(project, ref.crs)
        offset = read_offset(project)
        snapshot = output_dir / "baseline-roi.npz"
        atomic_npz(snapshot, rgba=rgba, corridor=corridor, surface_z=points[2].astype(np.float32))
        identity = {"schema": SCHEMA, "criticalInputs": binding, "grid": grid,
                    "corridorSha256": array_digest(corridor), "surfaceCropWindow": crop_window,
                    "surfaceInterpolation": "retained 2.5D surface bilinear; no adjustment",
                    "config": asdict(config), "detectorIdentity": detector_identity,
                    "snapshotSha256": sha256(snapshot)}
        identity["identitySha256"] = canonical_digest(identity)
        ranked = rank_corridor_sources(factory, data, points, corridor, offset, config.sample_cells)
        atomic_json(output_dir / "candidate-ranking.json", {"identitySha256": identity["identitySha256"], "allRankedShots": ranked,
                    "coverageIsEstimate": True, "rankingDomain": "all occupied corridor spatial cells, all checkpoint shots"})
        by_name = {r["sourceImage"]: r for r in ranked}
        if source_names is not None:
            if len(set(source_names)) != len(source_names) or any(name not in data["shots"] for name in source_names):
                raise ValueError("Explicit source names are duplicated or absent from reconstruction")
            chosen = [by_name[name] for name in source_names if name in by_name]
            if len(chosen) != len(source_names):
                raise ValueError("Explicit source has no sampled corridor support")
        else:
            chosen = [r for r in ranked if r["estimatedCorridorCoverage"] >= config.min_estimated_coverage][:config.max_sources]
        if not chosen:
            raise ValueError("No candidate has the required corridor coverage")
        sources, photo_binding = [], {}
        for index, rank in enumerate(chosen):
            name = rank["sourceImage"]
            shot = data["shots"][name]
            source_path = project / "opensfm/undistorted/images" / name
            photo_binding[str(source_path)] = sha256(source_path)
            camera_record = {"shot": shot, "camera": data["cameras"][shot["camera"]]}
            sy, sx = _sample_indices(corridor & np.isfinite(points[2]), config.sample_cells)
            agreement = native_projection_agreement(factory.get(name), shot, camera_record["camera"], offset, points[:, sy, sx])
            if not agreement["accepted"]:
                raise ValueError("Native perspective/Orthority projection disagreement: " + name)
            image, valid, cost = project_source(project, name, factory.get(name), shot, offset, points, surface, st, config)
            observations, detection_state, source_context = detect_source_bodies(project, name, factory.get(name), points, valid, detector, config)
            projected_detection_state = "not-run"
            if detector is not None and config.projected_detection_pass:
                supplemental, projected_detection_state = detect_projected_bodies(image, valid, detector, config)
                for observation in supplemental:
                    observation["detectionOrigin"] = "same-source-projected-supplement"
                observations.extend(supplemental)
            arrays = {"image": image, "valid": valid, "cost": cost, "body_mask": np.zeros(shape, bool)}
            if source_context is not None:
                arrays.update(source_context)
            vehicles = []
            for obs_index, observation in enumerate(observations):
                mask_key = f"vehicle_{obs_index}"
                mask = observation["mask"]
                arrays[mask_key] = mask
                arrays["body_mask"] |= mask
                vehicles.append({**{k: v for k, v in observation.items() if k != "mask"}, "maskKey": mask_key,
                                 "pixels": int(mask.sum()), "areaSquareMapUnits": float(mask.sum() * grid["pixelAreaSquareMapUnits"])})
            cache = output_dir / f"source-{index:03d}.npz"
            atomic_npz(cache, **arrays)
            metadata = {**rank, "schema": SCHEMA, "identitySha256": identity["identitySha256"], "grid": grid,
                        "sourceImagePath": str(source_path), "sourceImageSha256": photo_binding[str(source_path)],
                        "cameraParametersSha256": canonical_digest(camera_record), "cameraParameters": camera_record,
                        "nativeProjectionAgreement": agreement,
                        "cachePath": str(cache), "cacheSha256": sha256(cache),
                        "strictCorridorValidCoverage": float(valid[corridor].mean()),
                        "strictRoiValidCoverage": float(valid.mean()), "samplingSupport": "four valid source pixels in every RGB band",
                        "detectionState": detection_state, "projectedDetectionState": projected_detection_state, "vehicles": vehicles,
                        "zeroObjectsCertifiesEmptyRoad": False, "vehicleAppearanceProofRequired": True,
                        "geometryAcceptance": "not evaluated; independent static proof required"}
            atomic_json(cache.with_suffix(".json"), metadata)
            sources.append({**metadata, "sourceMetadataSha256": sha256(cache.with_suffix(".json"))})
            atomic_json(output_dir / "status.json", {"schema": SCHEMA, "state": "projecting", "completedSources": len(sources), "totalSources": len(chosen), "elapsedSeconds": time.monotonic() - started})
        verify_hash_binding({**binding, **photo_binding})
        manifest = {**identity, "sources": sources, "sourceInputHashes": photo_binding,
                    "state": "complete", "readOnlyInputsVerified": True,
                    "acceptance": "evidence-only; no mosaic change authorized", "elapsedSeconds": time.monotonic() - started}
        path = output_dir / "manifest.json"
        atomic_json(path, manifest)
        atomic_json(output_dir / "status.json", {"schema": SCHEMA, "state": "complete", "manifestSha256": sha256(path), "elapsedSeconds": time.monotonic() - started})
        return path
    except Exception as error:
        atomic_json(output_dir / "status.json", {"schema": SCHEMA, "state": "failed", "error": str(error), "elapsedSeconds": time.monotonic() - started})
        raise


def load_bound_source(manifest_path: Path, source_name: str, *, recheck_inputs: bool = True):
    """Load completed evidence only, rejecting stale inputs, caches, or camera records."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete":
        raise ValueError("Projection evidence is not a completed supported manifest")
    identity_keys = ("schema", "criticalInputs", "grid", "corridorSha256", "surfaceCropWindow", "surfaceInterpolation", "config", "detectorIdentity", "snapshotSha256")
    if canonical_digest({k: manifest[k] for k in identity_keys}) != manifest["identitySha256"]:
        raise ValueError("Projection identity metadata changed")
    if sha256(manifest_path.parent / "baseline-roi.npz") != manifest["snapshotSha256"]:
        raise ValueError("Preserved ROI snapshot changed")
    records = [r for r in manifest["sources"] if r["sourceImage"] == source_name]
    if len(records) != 1:
        raise ValueError("Source name is absent or ambiguous in evidence")
    record = records[0]
    if record["identitySha256"] != manifest["identitySha256"] or record["grid"] != manifest["grid"]:
        raise ValueError("Source cache uses a different grid or input identity")
    if canonical_digest(record["cameraParameters"]) != record["cameraParametersSha256"]:
        raise ValueError("Source camera parameters changed")
    if recheck_inputs:
        verify_hash_binding({**manifest["criticalInputs"], record["sourceImagePath"]: record["sourceImageSha256"]})
        reconstruction_paths = [Path(p) for p in manifest["criticalInputs"] if Path(p).name == "reconstruction.json"]
        if len(reconstruction_paths) != 1:
            raise ValueError("Projection identity has no unique retained camera reconstruction")
        reconstruction_path = reconstruction_paths[0]
        actual_data = json.loads(reconstruction_path.read_text(encoding="utf-8"))
        if len(actual_data) != 1 or source_name not in actual_data[0]["shots"]:
            raise ValueError("Bound source is absent from the retained reconstruction")
        actual_shot = actual_data[0]["shots"][source_name]
        actual_camera = actual_data[0]["cameras"][actual_shot["camera"]]
        if canonical_digest({"shot": actual_shot, "camera": actual_camera}) != record["cameraParametersSha256"]:
            raise ValueError("Source camera record does not match the retained reconstruction")
        expected_source = reconstruction_path.parent / "images" / source_name
        if Path(record["sourceImagePath"]).resolve() != expected_source.resolve():
            raise ValueError("Source image path does not belong to the retained camera checkpoint")
    if manifest["sourceInputHashes"].get(record["sourceImagePath"]) != record["sourceImageSha256"]:
        raise ValueError("Source image record does not match the completed input hash binding")
    cache = Path(record["cachePath"])
    if cache.parent.resolve() != manifest_path.parent.resolve() or sha256(cache) != record["cacheSha256"]:
        raise ValueError("Source cache path or bytes changed")
    source_metadata = cache.with_suffix(".json")
    if sha256(source_metadata) != record["sourceMetadataSha256"]:
        raise ValueError("Source metadata bytes changed")
    if json.loads(source_metadata.read_text(encoding="utf-8")) != {k: v for k, v in record.items() if k != "sourceMetadataSha256"}:
        raise ValueError("Source metadata does not match the completed manifest")
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    shape = (manifest["grid"]["height"], manifest["grid"]["width"])
    if arrays["image"].shape != (*shape, 3) or arrays["image"].dtype != np.uint8:
        raise ValueError("Source image cache has invalid shape or dtype")
    if any(arrays[key].shape != shape for key in ("valid", "cost", "body_mask")):
        raise ValueError("Source cache masks/cost do not match the grid")
    if arrays["valid"].dtype != bool or arrays["body_mask"].dtype != bool or arrays["cost"].dtype != np.float32:
        raise ValueError("Source cache mask/cost dtypes changed")
    if not np.isfinite(arrays["cost"]).all() or np.any(arrays["cost"][~arrays["valid"]] < 1e6):
        raise ValueError("Invalid source samples can be selected at finite cost")
    for observation in record["vehicles"]:
        key = observation["maskKey"]
        if key not in arrays or arrays[key].shape != shape or arrays[key].dtype != bool:
            raise ValueError("Vehicle mask does not match its evidence grid")
    union = np.zeros(shape, bool)
    for observation in record["vehicles"]:
        union |= arrays[observation["maskKey"]]
    if not np.array_equal(union, arrays["body_mask"]):
        raise ValueError("Body protection mask differs from the recorded observations")
    return arrays, record, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("project", "grid", "surface", "corridor-mask", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--window", type=int, nargs=4, required=True, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sources", type=int, default=16)
    parser.add_argument("--min-estimated-coverage", type=float, default=0.35)
    args = parser.parse_args()
    config = ProjectionConfig(max_sources=args.max_sources, min_estimated_coverage=args.min_estimated_coverage)
    mask = cv2.imread(str(args.corridor_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Could not read corridor mask")
    detector, model_identity = None, None
    if args.models:
        model_files = sorted(p for p in args.models.rglob("*") if p.is_file())
        model_identity = {"modelFiles": {str(p.resolve()): sha256(p) for p in model_files}, "detectorConfig": asdict(Config()), "maskCoordinateFrame": "real-source same-grid projected ROI"}
        detector = Detector(args.models, args.device, Config())
    result = build_corridor_evidence(args.project, args.grid, args.surface, args.output, args.window, mask > 0,
                                     detector=detector, detector_identity=model_identity, config=config)
    print(str(result), flush=True)


if __name__ == "__main__":
    main()
