"""OGS2: offline, object-coherent orthomosaics from a preserved ODM checkpoint.

This is an opt-in post-reconstruction engine. It never overwrites the checkpoint.
Orthority provides the calibrated projection, GroundingDINO and SAM 2 the masks.
The object/source constraints, visibility checks and bounded mosaic are ours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage


@dataclass(frozen=True)
class Config:
    protect_objects: bool = True
    candidates: int = 8
    tile_size: int = 768
    halo: int = 160
    surface_scale: int = 2
    detector_threshold: float = 0.25
    text_threshold: float = 0.20
    min_sam_score: float = 0.8
    min_box_support: float = 0.9
    max_group_to_core_ratio: float = 4.0
    object_padding_m: float = 0.6
    min_object_pixels: int = 16
    visibility_tolerance_m: float = 0.35
    seam_scale: int = 4
    cpu_threads: int = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_offset(project: Path) -> np.ndarray:
    lines = (project / "odm_georeferencing/odm_georeferencing_model_geo.txt").read_text().splitlines()
    xy = np.asarray([float(v) for v in lines[1].split()])
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Invalid checkpoint coordinate offset")
    return np.r_[xy, 0.0]


def cameras_from_checkpoint(project: Path, crs: rio.CRS):
    from orthority.factory import FrameCameras
    from orthority.param_io import OsfmReader

    file = project / "opensfm/undistorted/reconstruction.json"
    reconstruction = json.loads(file.read_text())
    if len(reconstruction) != 1:
        raise ValueError("Only a single connected reconstruction is supported")
    data = reconstruction[0]
    reader = OsfmReader(file, crs=crs)
    intrinsic, extrinsic = reader.read_int_param(), reader.read_ext_param()
    offset = read_offset(project)
    for name, params in extrinsic.items():
        shot = data["shots"][name]
        rotation = cv2.Rodrigues(np.array(shot["rotation"], dtype=np.float64))[0]
        # ODM adds its explicit map offset to the local reconstruction. Do not
        # introduce another ENU/UTM rotation or recompute GPS-only camera poses.
        params["xyz"] = tuple(-rotation.T @ np.array(shot["translation"]) + offset)
    factory = FrameCameras(intrinsic, extrinsic, io_kwargs={"crs": crs})
    for name, shot in data["shots"].items():
        params = data["cameras"][shot["camera"]]
        if params["projection_type"] != "perspective" or any(params.get(k, 0) != 0 for k in ("k1", "k2")):
            raise ValueError("Expected matched undistorted perspective images")
        path = project / "opensfm/undistorted/images" / name
        with rio.open(path) as src:
            if (src.width, src.height) != factory.get(name).im_size:
                raise ValueError("Camera/image dimension mismatch: " + name)
    return factory, data


def _rasterize(vertices, faces, height, width):
    """Topmost barycentric triangle rasterization; no fabricated hole filling."""
    out = np.full((height, width), -np.inf, dtype=np.float32)
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        x0 = max(0, int(math.ceil(min(a[0], b[0], c[0]))))
        x1 = min(width - 1, int(math.floor(max(a[0], b[0], c[0]))))
        y0 = max(0, int(math.ceil(min(a[1], b[1], c[1]))))
        y1 = min(height - 1, int(math.floor(max(a[1], b[1], c[1]))))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
                v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
                if u >= -1e-5 and v >= -1e-5 and u + v <= 1.00001:
                    z = u * a[2] + v * b[2] + (1 - u - v) * c[2]
                    out[y, x] = max(out[y, x], z)
    return out


def export_surface(project: Path, grid: Path, output: Path, scale: int) -> dict:
    from numba import njit
    from plyfile import PlyData

    mesh = project / "odm_meshing/odm_25dmesh.ply"
    expected = {"meshSha256": sha256(mesh), "gridSha256": sha256(grid), "scale": scale}
    meta = output.with_suffix(".json")
    if output.exists() and meta.exists() and json.loads(meta.read_text()) == expected:
        return expected
    ply = PlyData.read(mesh)
    v = ply["vertex"].data
    vertices = np.column_stack([v[k] for k in ("x", "y", "z")]).astype(np.float64)
    faces = np.vstack(ply["face"].data["vertex_indices"]).astype(np.int32)
    with rio.open(grid) as src:
        if not src.crs or not src.crs.is_projected or src.transform.b or src.transform.d:
            raise ValueError("Require a north-up projected reference grid")
        width, height = math.ceil(src.width / scale), math.ceil(src.height / scale)
        transform = src.transform * Affine.scale(scale)
        profile = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=src.crs,
            transform=transform,
            nodata=np.nan,
            compress="DEFLATE",
            tiled=True,
        )
    vertices += read_offset(project)
    inverse = ~transform
    x, y = inverse * (vertices[:, 0], vertices[:, 1])
    vertices[:, 0], vertices[:, 1] = x - 0.5, y - 0.5
    surface = njit(_rasterize)(vertices, faces, height, width)
    surface[~np.isfinite(surface)] = np.nan
    with rio.open(output, "w", **profile) as dst:
        dst.write(surface, 1)
    save_json(meta, expected)
    return expected


def surface_points(surface, transform, window: Window, grid_transform: Affine):
    h, w = int(window.height), int(window.width)
    x, y = grid_transform * np.meshgrid(
        np.arange(w) + window.col_off + 0.5, np.arange(h) + window.row_off + 0.5
    )
    columns, rows = (~transform) * (x, y)
    z = ndimage.map_coordinates(
        surface, [rows - 0.5, columns - 0.5], order=1, mode="constant", cval=np.nan, prefilter=False
    )
    return np.stack([x, y, z])


def visibility_mask(points, camera_position, surface, transform, tolerance):
    """Ray/surface test at distance-adaptive samples; unknown terrain is not visible."""
    from model_engine.ogs2_visibility import ray_visible

    # Probe the first half of the ray densely near the target, where aerial
    # occlusion normally occurs. A height-field cannot certify overhang geometry.
    step_m = max(abs(transform.a), abs(transform.e))
    return ray_visible(
        points,
        np.asarray(camera_position),
        surface,
        np.array(list(~transform)[:6]),
        np.geomspace(step_m, max(step_m * 2, 40), 24),
        tolerance,
        step_m,
    )


def reliable_mask(mask, box, sam_score, config):
    """Reject uncertain masks and masks leaking outside the object proposal.

    Do not clip a leaked mask and then claim it is a complete object.
    SAM's quality score is a filter, not a guarantee of semantic correctness.
    """
    if sam_score < config.min_sam_score or mask.sum() < config.min_object_pixels:
        return False
    x0, y0, x1, y1 = box
    left, top = max(0, math.floor(x0)), max(0, math.floor(y0))
    right, bottom = min(mask.shape[1], math.ceil(x1)), min(mask.shape[0], math.ceil(y1))
    inside = mask[top:bottom, left:right].sum() if right > left and bottom > top else 0
    return inside / mask.sum() >= config.min_box_support


class Detector:
    def __init__(self, models: Path, device: str, config: Config):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device, self.config = torch, device, config
        torch.set_num_threads(config.cpu_threads)
        if str(device).startswith("cuda"):
            selected_device = torch.device(device)
            if selected_device.index is None:
                selected_device = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(0.12, device=selected_device)
        self.processor = AutoProcessor.from_pretrained(models / "grounding-dino-tiny", local_files_only=True)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                models / "grounding-dino-tiny", local_files_only=True, disable_custom_kernels=True
            )
            .to(device)
            .eval()
        )
        sam = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(models / "sam2.1_hiera_tiny.pt"),
            device=device,
            apply_postprocessing=False,
        )
        self.sam = SAM2ImagePredictor(sam)

    def masks(self, image: np.ndarray) -> list[dict]:
        from PIL import Image

        torch = self.torch
        prompt = "car. truck. bus. excavator. crane. construction vehicle."
        inputs = self.processor(images=Image.fromarray(image), text=prompt, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.config.detector_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=[image.shape[:2]],
            )[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            if len(boxes) == 0:
                return []
            self.sam.set_image(image)
            masks, sam_scores, _ = self.sam.predict(box=boxes, multimask_output=False)
        masks = masks.reshape((-1, *image.shape[:2]))
        records = []
        for mask, box, confidence, sam_score in zip(masks, boxes, scores, np.ravel(sam_scores)):
            mask = mask.astype(bool)
            if not reliable_mask(mask, box, float(sam_score), self.config):
                continue
            ys, xs = np.where(mask)
            margin = min(xs.min(), ys.min(), image.shape[1] - 1 - xs.max(), image.shape[0] - 1 - ys.max())
            # Objects touching a source crop are not assumed to be complete.
            records.append(
                {
                    "mask": mask,
                    "box": box,
                    "confidence": float(confidence),
                    "samConfidence": float(sam_score),
                    "complete": bool(margin >= 5),
                }
            )
        return records


def group_masks(masks: list[np.ndarray], padding: int) -> list[np.ndarray]:
    """Merge touching/overlapping projected observations into indivisible regions."""
    if not masks:
        return []
    union = np.zeros(masks[0].shape, bool)
    for mask in masks:
        if mask.any():
            union |= ndimage.binary_dilation(mask, iterations=max(1, padding))
    labels, n = ndimage.label(union)
    return [labels == i for i in range(1, n + 1)]


def lock_objects(
    labels: np.ndarray,
    regions: list[np.ndarray],
    costs: np.ndarray,
    valid: np.ndarray,
    observations: list[dict],
    config: Config,
) -> list[dict]:
    records = []
    for region in regions:
        # A source must observe a complete object; a low-cost background view
        # without the object is never credited with restoring a vehicle.
        eligible = []
        rejection_counts = {"noCompleteObject": 0, "occludedObjectCore": 0, "ambiguousGroup": 0}
        for index in range(len(costs)):
            obs = [
                o
                for o in observations
                if o["view"] == index
                and o["complete"]
                and (o["mask"] & region).sum() >= config.min_object_pixels
            ]
            if not obs:
                rejection_counts["noCompleteObject"] += 1
                continue
            core = np.zeros(region.shape, bool)
            for item in obs:
                core |= item["mask"] & region
            # One tiny observed vehicle cannot authorize replacing a large
            # merged region. Moving-object trails and segmentation leaks must
            # fall back rather than silently erasing other observations.
            if region.sum() > config.max_group_to_core_ratio * core.sum():
                rejection_counts["ambiguousGroup"] += 1
                continue
            coverage = float(valid[index][core].mean())
            if not valid[index][core].all():
                rejection_counts["occludedObjectCore"] += 1
                continue
            eligible.append((float(np.mean(costs[index][core])), index, coverage, int(core.sum())))
        if not eligible:
            labels[region] = -1
            records.append(
                {"state": "baseline-fallback", "pixels": int(region.sum()), "rejections": rejection_counts}
            )
            continue
        cost, selected, coverage, core_pixels = min(eligible)
        labels[region] = -1
        labels[region & valid[selected]] = selected
        records.append(
            {
                "state": "locked",
                "view": selected,
                "pixels": int(region.sum()),
                "coverage": coverage,
                "meanCost": cost,
                "objectCorePixels": core_pixels,
                "paddingFallbackPixels": int((region & ~valid[selected]).sum()),
            }
        )
    return records


def graphcut_labels(images: np.ndarray, costs: np.ndarray, valid: np.ndarray, config: Config) -> np.ndarray:
    """Color/gradient seam optimization followed by hard object constraints."""
    n, h, w = costs.shape
    sh, sw = max(1, h // config.seam_scale), max(1, w // config.seam_scale)
    scaled_costs = np.stack([cv2.resize(c, (sw, sh), interpolation=cv2.INTER_AREA) for c in costs])
    initial = np.argmin(scaled_costs, axis=0)
    masks = []
    floats = []
    for i in range(n):
        mask = (initial == i).astype(np.uint8) * 255
        # Expand assignment borders to create limited overlap for graph cuts.
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        mask &= cv2.resize(valid[i].astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
        masks.append(cv2.UMat(mask))
        floats.append(cv2.UMat(cv2.resize(images[i], (sw, sh)).astype(np.float32)))
    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    try:
        finder.find(floats, [(0, 0)] * n, masks)
        allowed = np.stack([m.get() > 0 for m in masks])
        scaled_costs[~allowed] = 1e6
        initial = np.argmin(scaled_costs, axis=0)
    except cv2.error as error:
        # This is an explicit numerical fallback, not an undetected success.
        print("graphcut fallback:", str(error).splitlines()[0], flush=True)
    labels = cv2.resize(initial.astype(np.int16), (w, h), interpolation=cv2.INTER_NEAREST)
    picked = np.take_along_axis(valid, labels[None], axis=0)[0]
    labels[~picked] = -1
    return labels


def candidate_views(factory, data, center, count):
    scores = []
    for name in sorted(data["shots"]):
        camera = factory.get(name)
        pixel = camera.world_to_pixel(center.reshape(3, 1)).ravel()
        w, h = camera.im_size
        if 0 <= pixel[0] < w and 0 <= pixel[1] < h and camera.pos[2] > center[2]:
            height = camera.pos[2] - center[2]
            distance = np.linalg.norm(np.asarray(camera.pos[:2]) - center[:2]) / height
            scores.append((distance, name))
    return [name for _, name in sorted(scores)[:count]]


def project_view(project, name, camera, points, surface, surface_transform, detector, config):
    h, w = points.shape[1:]
    coordinates = camera.world_to_pixel(points.reshape(3, -1)).reshape(2, h, w).astype(np.float32)
    finite = np.isfinite(coordinates).all(axis=0) & np.isfinite(points[2])
    iw, ih = camera.im_size
    inside = finite & (coordinates[0] >= 1) & (coordinates[0] < iw - 2)
    inside &= (coordinates[1] >= 1) & (coordinates[1] < ih - 2)
    if not inside.any():
        return None
    x0 = max(0, int(np.floor(np.min(coordinates[0][inside]))) - 48)
    y0 = max(0, int(np.floor(np.min(coordinates[1][inside]))) - 48)
    x1 = min(iw, int(np.ceil(np.max(coordinates[0][inside]))) + 49)
    y1 = min(ih, int(np.ceil(np.max(coordinates[1][inside]))) + 49)
    with rio.open(project / "opensfm/undistorted/images" / name) as src:
        image = src.read([1, 2, 3], window=Window(x0, y0, x1 - x0, y1 - y0)).transpose(1, 2, 0)
    maps = coordinates - np.array([x0, y0], dtype=np.float32)[:, None, None]
    safe_maps = np.nan_to_num(maps, nan=-1e6)
    rgb = cv2.remap(image, safe_maps[0], safe_maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    visible = visibility_mask(points, camera.pos, surface, surface_transform, config.visibility_tolerance_m)
    valid = inside & visible
    # Penalize strongly tilted surface-to-photo mappings, in addition to the
    # camera's distance from nadir. This does not repair an incorrect surface.
    du_y, du_x = np.gradient(coordinates[0])
    dv_y, dv_x = np.gradient(coordinates[1])
    determinant = np.abs(du_x * dv_y - du_y * dv_x)
    mapping_condition = (du_x**2 + du_y**2 + dv_x**2 + dv_y**2) / np.maximum(determinant, 1e-4)
    distance = np.hypot(points[0] - camera.pos[0], points[1] - camera.pos[1])
    height = camera.pos[2] - points[2]
    cost = (distance / np.maximum(height, 1)) ** 2 + 0.1 * np.maximum(mapping_condition - 2, 0)
    cost[~valid] = 1e6
    observations = []
    for observation in detector.masks(image) if detector is not None else []:
        mask = (
            cv2.remap(
                observation["mask"].astype(np.uint8), safe_maps[0], safe_maps[1], cv2.INTER_NEAREST
            ).astype(bool)
            & inside
        )
        if mask.sum() >= config.min_object_pixels:
            observations.append({**observation, "mask": mask})
    return rgb, valid, np.nan_to_num(cost, nan=1e6).astype(np.float32), observations


def run(
    project: Path, output: Path, models: Path, config: Config, device: str, bounds: list[float] | None = None
):
    started = time.monotonic()
    project, output = project.resolve(), output.resolve()
    if output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError("Output must be separate from the preserved checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "ogs2_orthophoto.partial.tif"
    if final.exists() or (output / "ogs2_orthophoto.tif").exists():
        raise FileExistsError("Refuse to overwrite an existing experiment")
    baseline = project / "odm_orthophoto/odm_orthophoto.tif"
    critical = [
        baseline,
        project / "opensfm/undistorted/reconstruction.json",
        project / "odm_meshing/odm_25dmesh.ply",
    ]
    before = {str(p): sha256(p) for p in critical}
    save_json(output / "status.json", {"state": "preparing", "config": asdict(config)})
    cv2.setNumThreads(config.cpu_threads)
    cv2.ocl.setUseOpenCL(False)
    surface_path = output.parent / "ogs2-surface.tif"
    surface_info = export_surface(project, baseline, surface_path, config.surface_scale)
    with rio.open(surface_path) as src:
        surface = src.read(1)
        surface_transform = src.transform
    with rio.open(baseline) as ref:
        factory, data = cameras_from_checkpoint(project, ref.crs)
        detector = Detector(models, device, config) if config.protect_objects else None
        if bounds:
            raw = rio.windows.from_bounds(*bounds, transform=ref.transform)
            window = raw.round_offsets().round_lengths().intersection(Window(0, 0, ref.width, ref.height))
        else:
            window = Window(0, 0, ref.width, ref.height)
        width, height = int(window.width), int(window.height)
        transform = ref.window_transform(window)
        profile = ref.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform,
            count=4,
            dtype="uint8",
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=None,
        )
        source_profile = {**profile, "count": 1, "dtype": "uint16", "nodata": 0}
        names = sorted(data["shots"])
        source_ids = {name: i + 1 for i, name in enumerate(names)}
        tiles = []
        observation_database = {}
        protected = np.zeros((height, width), dtype=bool)
        total_changed, total_locked = 0, 0
        with (
            rio.open(final, "w", **profile) as dst,
            rio.open(output / "source-index.tif", "w", **source_profile) as source_dst,
        ):
            dst.colorinterp = (
                rio.enums.ColorInterp.red,
                rio.enums.ColorInterp.green,
                rio.enums.ColorInterp.blue,
                rio.enums.ColorInterp.alpha,
            )
            for row in range(0, height, config.tile_size):
                for col in range(0, width, config.tile_size):
                    core = Window(
                        col, row, min(config.tile_size, width - col), min(config.tile_size, height - row)
                    )
                    x0, y0 = max(0, col - config.halo), max(0, row - config.halo)
                    x1, y1 = (
                        min(width, col + core.width + config.halo),
                        min(height, row + core.height + config.halo),
                    )
                    ext = Window(x0, y0, x1 - x0, y1 - y0)
                    points = surface_points(surface, surface_transform, ext, transform)
                    read_window = Window(window.col_off + x0, window.row_off + y0, ext.width, ext.height)
                    old = ref.read(window=read_window)
                    if old.shape[0] < 4:
                        old = np.concatenate([old[:3], ref.dataset_mask(window=read_window)[None]])
                    center = np.nanmedian(points.reshape(3, -1), axis=1)
                    chosen = (
                        candidate_views(factory, data, center, config.candidates)
                        if np.isfinite(center).all()
                        else []
                    )
                    images, valids, costs, observations, used = [], [], [], [], []
                    for name in chosen:
                        value = project_view(
                            project,
                            name,
                            factory.get(name),
                            points,
                            surface,
                            surface_transform,
                            detector,
                            config,
                        )
                        if value is None:
                            continue
                        image, valid, cost, obs = value
                        index = len(images)
                        images.append(image)
                        valids.append(valid)
                        costs.append(cost)
                        used.append(name)
                        observations.extend([{**o, "view": index} for o in obs])
                        for observation in obs:
                            oy, ox = np.where(observation["mask"])
                            if len(ox) == 0:
                                continue
                            left, top, right, bottom = ox.min(), oy.min(), ox.max() + 1, oy.max() + 1
                            observation_database.setdefault(name, []).append(
                                {
                                    "sourceImage": name,
                                    "complete": observation["complete"],
                                    "confidence": observation["confidence"],
                                    "bounds": [
                                        int(x0 + left),
                                        int(y0 + top),
                                        int(x0 + right),
                                        int(y0 + bottom),
                                    ],
                                    "mask": observation["mask"][top:bottom, left:right].copy(),
                                }
                            )
                    result = old[:4].transpose(1, 2, 0).copy()
                    sources = np.zeros(result.shape[:2], np.uint16)
                    object_records = []
                    if images:
                        images, valids, costs = np.stack(images), np.stack(valids), np.stack(costs)
                        labels = graphcut_labels(images, costs, valids, config)
                        pad = max(1, round(config.object_padding_m / abs(transform.a)))
                        regions = group_masks([o["mask"] for o in observations], pad)
                        for region in regions:
                            protected[int(y0) : int(y1), int(x0) : int(x1)] |= region
                        object_records = lock_objects(labels, regions, costs, valids, observations, config)
                        for item in object_records:
                            if item["state"] == "locked":
                                name = used[item["view"]]
                                item["sourceImage"] = name
                                item["captureTime"] = data["shots"][name].get("capture_time")
                        # Preserve baseline coverage and leave unsupported surfaces unchanged.
                        labels[old[3] == 0] = -1
                        for index, name in enumerate(used):
                            mask = labels == index
                            result[mask, :3] = images[index][mask]
                            sources[mask] = source_ids[name]
                    iy, ix = row - y0, col - x0
                    cut = np.s_[int(iy) : int(iy + core.height), int(ix) : int(ix + core.width)]
                    dst.write(result[cut].transpose(2, 0, 1), window=core)
                    source_dst.write(sources[cut], 1, window=core)
                    changed = int(
                        np.any(result[cut][..., :3] != old.transpose(1, 2, 0)[cut][..., :3], axis=2).sum()
                    )
                    total_changed += changed
                    total_locked += sum(o["state"] == "locked" for o in object_records)
                    entry = {
                        "window": [col, row, int(core.width), int(core.height)],
                        "sources": used,
                        "objects": object_records,
                        "changedPixels": changed,
                    }
                    tiles.append(entry)
                    save_json(output / "tiles.json", tiles)
                    save_json(
                        output / "status.json",
                        {
                            "state": "running",
                            "tiles": len(tiles),
                            "elapsedSeconds": time.monotonic() - started,
                        },
                    )
                    print(
                        "tile",
                        len(tiles),
                        "views",
                        len(used),
                        "objects",
                        len(object_records),
                        "locked",
                        sum(o["state"] == "locked" for o in object_records),
                        flush=True,
                    )
        # Reconcile object source choices globally after tile seams. Otherwise
        # a vehicle straddling two tiles could still be split between photos.
        groups, group_count = ndimage.label(protected)
        slices = ndimage.find_objects(groups)
        global_objects = []
        with rio.open(final, "r+") as dst, rio.open(output / "source-index.tif", "r+") as source_dst:
            for group_id, box in enumerate(slices, 1):
                if box is None:
                    continue
                ys, xs = box
                pad = max(16, round(2 / abs(transform.a)))
                x0, y0 = max(0, xs.start - pad), max(0, ys.start - pad)
                x1, y1 = min(width, xs.stop + pad), min(height, ys.stop + pad)
                region = groups[y0:y1, x0:x1] == group_id
                window_here = Window(x0, y0, x1 - x0, y1 - y0)
                if region.sum() < config.min_object_pixels:
                    continue
                if (x1 - x0) * (y1 - y0) > 4_000_000:
                    global_objects.append(
                        {
                            "state": "baseline-fallback",
                            "reason": "ambiguous-large-group",
                            "pixels": int(region.sum()),
                        }
                    )
                    old_region = ref.read(
                        window=Window(window.col_off + x0, window.row_off + y0, x1 - x0, y1 - y0)
                    )
                    current = dst.read(window=window_here)
                    current[:, region] = old_region[:, region]
                    dst.write(current, window=window_here)
                    ids = source_dst.read(1, window=window_here)
                    ids[region] = 0
                    source_dst.write(ids, 1, window=window_here)
                    continue
                points = surface_points(surface, surface_transform, window_here, transform)
                center = np.nanmedian(points[:, region], axis=1)
                chosen = candidate_views(factory, data, center, config.candidates)
                images, valids, costs, observations, used = [], [], [], [], []
                for name in chosen:
                    value = project_view(
                        project, name, factory.get(name), points, surface, surface_transform, None, config
                    )
                    if value is None:
                        continue
                    image, valid, cost, obs = value
                    index = len(images)
                    images.append(image)
                    valids.append(valid)
                    costs.append(cost)
                    used.append(name)
                    for cached in observation_database.get(name, []):
                        left, top, right, bottom = cached["bounds"]
                        overlap_left, overlap_top = max(left, x0), max(top, y0)
                        overlap_right, overlap_bottom = min(right, x1), min(bottom, y1)
                        if overlap_left >= overlap_right or overlap_top >= overlap_bottom:
                            continue
                        mask = np.zeros(region.shape, bool)
                        mask[
                            overlap_top - y0 : overlap_bottom - y0, overlap_left - x0 : overlap_right - x0
                        ] = cached["mask"][
                            overlap_top - top : overlap_bottom - top,
                            overlap_left - left : overlap_right - left,
                        ]
                        observations.append(
                            {
                                "view": index,
                                "complete": cached["complete"],
                                "confidence": cached["confidence"],
                                "mask": mask,
                            }
                        )
                selected = np.full(region.shape, -1, dtype=np.int16)
                object_records = (
                    lock_objects(
                        selected, [region], np.asarray(costs), np.asarray(valids), observations, config
                    )
                    if images
                    else [{"state": "baseline-fallback", "reason": "no-source", "pixels": int(region.sum())}]
                )
                record = {**object_records[0], "group": group_id, "window": [x0, y0, x1 - x0, y1 - y0]}
                current = dst.read(window=window_here)
                ids = source_dst.read(1, window=window_here)
                old_region = ref.read(
                    window=Window(window.col_off + x0, window.row_off + y0, x1 - x0, y1 - y0)
                )
                if record["state"] == "locked":
                    index = record["view"]
                    name = used[index]
                    permitted = region & (selected == index) & (old_region[3] != 0)
                    rejected_padding = region & (selected != index)
                    current[:, rejected_padding] = old_region[:, rejected_padding]
                    ids[rejected_padding] = 0
                    current[:3, permitted] = images[index][permitted].T
                    ids[permitted] = source_ids[name]
                    record["sourceImage"] = name
                    record["captureTime"] = data["shots"][name].get("capture_time")
                    record["sourceConfidence"] = max(
                        o["confidence"]
                        for o in observations
                        if o["view"] == index and (o["mask"] & region).any()
                    )
                else:
                    current[:, region] = old_region[:, region]
                    ids[region] = 0
                dst.write(current, window=window_here)
                source_dst.write(ids, 1, window=window_here)
                global_objects.append(record)
                save_json(output / "objects.json", global_objects)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
        save_json(output / "objects.json", global_objects)
        with rio.open(output / "protected-objects.tif", "w", **{**source_profile, "dtype": "uint8"}) as dst:
            dst.write(protected.astype(np.uint8), 1)
        after = {str(p): sha256(p) for p in critical}
        if before != after:
            raise RuntimeError("Preserved checkpoint changed during execution")
        completed = output / "ogs2_orthophoto.tif"
        final.replace(completed)
        report = {
            "schemaVersion": "ogs2-run/v1",
            "engineSha256": sha256(Path(__file__)),
            "visibilitySha256": sha256(Path(__file__).with_name("ogs2_visibility.py")),
            "config": asdict(config),
            "project": str(project),
            "surface": surface_info,
            "inputHashes": before,
            "baselineUnchanged": True,
            "imageCount": len(names),
            "output": str(completed),
            "grid": {"crs": str(ref.crs), "transform": list(transform), "width": width, "height": height},
            "changedPixelsBeforeGlobalReconciliation": total_changed,
            "lockedObjectObservations": total_locked,
            "protectedSpatialGroups": group_count,
            "globallyLockedGroups": sum(o["state"] == "locked" for o in global_objects),
            "elapsedSeconds": time.monotonic() - started,
            "sourceIndex": {str(v): k for k, v in source_ids.items()},
            "limitations": [
                "Objects may move between exposures; source time is explicit.",
                "A height field cannot represent crane overhangs or recover unseen structures.",
                "Surface-ray visibility is an approximation; geometry is unchanged.",
                "No generated pixels, no free image warp, no coordinate fit to DJI.",
                "Spatial groups can merge observations across times; counts are not unique vehicles.",
            ],
        }
        save_json(output / "report.json", report)
        save_json(output / "status.json", {"state": "success", "elapsedSeconds": report["elapsedSeconds"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--bounds", nargs=4, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--without-object-protection", action="store_true", help="Projection-only ablation")
    args = parser.parse_args()
    config = Config(candidates=args.candidates, protect_objects=not args.without_object_protection)
    if args.candidates < 1:
        parser.error("Candidates must be positive")
    try:
        run(args.project, args.output, args.models, config, args.device, args.bounds)
    except Exception as error:
        if args.output.exists():
            save_json(
                args.output / "status.json",
                {"state": "failed", "error": type(error).__name__ + ": " + str(error)},
            )
        raise


if __name__ == "__main__":
    main()
