"""Independent road measurements across the entire declared highway."""

import functools
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import Config, Detector, cameras_from_checkpoint, read_offset
from model_engine.ogs2_corridor_projection import bilinear_support
from model_engine.ogs2_road_surface import RoadSurfaceConfig, fit_road_surface
from PIL import Image
from rasterio.transform import Affine

r = Path("/volume/wzhang/mesh-research-workspace")
job = r / "runs/mangrove-077-20261013"
base = job / "highway-full-20261014-01"
project = job / "datasets/project"
out = base / "geometry-04"
out.mkdir(exist_ok=False)
plan = json.loads((base / "camera-corridor-plan.json").read_text())
tr0 = Affine(*plan["transform"])
maskdata = np.load(base / "road-mask-08/road-mask.npz")
roadlow = maskdata["road"]
top = maskdata["top"]
bottom = maskdata["bottom"]
mid = maskdata["median"]
with rio.open(job / "vehicle-ogs2/ogs2_orthophoto.tif") as grid:
    factory, data = cameras_from_checkpoint(project, grid.crs)
offset = read_offset(project)
provenance = dict(
    reconstruction_sha256=hashlib.sha256(
        (project / "opensfm/undistorted/reconstruction.json").read_bytes()
    ).hexdigest(),
    image_coordinate_frame="undistorted-photo",
    source="new-static-road-photo-triangulation",
)
detector = Detector(r / "runtime/vehicle-models", "cuda:0", Config())
sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.007, edgeThreshold=12)
bf = cv2.BFMatcher()
reports = []


def dynamic_guard(im):
    inputs = detector.processor(
        images=Image.fromarray(im), text="car. truck. bus. van. construction vehicle.", return_tensors="pt"
    ).to(detector.device)
    with detector.torch.inference_mode():
        outputs = detector.model(**inputs)
    result = detector.processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.20, text_threshold=0.18, target_sizes=[im.shape[:2]]
    )[0]
    boxes = result["boxes"].cpu().numpy()
    guard = np.zeros(im.shape[:2], np.uint8)
    for left_col, t, rr, b in boxes:
        pad = 20
        guard[
            max(0, int(t) - pad) : min(im.shape[0], int(b) + pad),
            max(0, int(left_col) - pad) : min(im.shape[1], int(rr) + pad),
        ] = 1
    return guard > 0, boxes.tolist()


def matrix(name):
    shot = data["shots"][name]
    c = data["cameras"][shot["camera"]]
    f = c["focal"] * max(c["width"], c["height"])
    K = np.array([[f, 0, (c["width"] - 1) / 2], [0, f, (c["height"] - 1) / 2], [0, 0, 1.0]])
    return K @ np.column_stack([cv2.Rodrigues(np.array(shot["rotation"]))[0], shot["translation"]])


projectors = {
    name: functools.partial(lambda n, p: factory.get(n).world_to_pixel(np.asarray(p).T).T, name)
    for name in data["shots"]
}
for segment, x0 in enumerate(range(1200, 24000, 1400)):
    start = time.time()
    x1 = min(24945, x0 + 2200)
    low0 = x0 // 4
    low1 = min(len(mid), int(np.ceil(x1 / 4)))
    y0 = max(0, int(top[low0:low1].min() * 4) - 100)
    y1 = min(4500, int(bottom[low0:low1].max() * 4) + 100)
    w = (x1 - x0) // 2
    h = (y1 - y0) // 2
    tr = tr0 * Affine.translation(x0, y0) * Affine.scale(2)
    yy, xx = np.mgrid[:h, :w]
    X, Y = tr * (xx + 0.5, yy + 0.5)
    P = np.stack([X, Y, np.full((h, w), 3.0)])
    lowx = (x0 + xx * 2) // 4
    lowy = (y0 + yy * 2) // 4
    road = roadlow[lowy, lowx]
    div = mid[lowx]
    north = maskdata["north"][lowy, lowx] & (lowy > div - 125)
    south = maskdata["south"][lowy, lowx] & (lowy < div + 150)
    measurement_road = north | south
    centre = np.array([*tr0 * ((x0 + x1) / 2, float(mid[(low0 + low1) // 2]) * 4), 3.0])
    ranked = []
    for c in plan["allCameras"]:
        if c["index"] > 95:
            continue
        cam = factory.get(c["name"])
        uv = cam.world_to_pixel(centre[:, None]).ravel()
        iw, ih = cam.im_size
        if 35 < uv[0] < iw - 35 and 35 < uv[1] < ih - 35:
            ranked.append((np.linalg.norm(np.asarray(cam.pos[:2]) - centre[:2]), c))
    # Independent baselines, all from the same traversal. A later view is used
    # only if it actually covers the road; camera XY proximity is insufficient.
    chosen = []
    for _, c in sorted(ranked, key=lambda p: p[0]):
        if all(abs(c["index"] - q["index"]) >= 3 for q in chosen):
            chosen.append(c)
        if len(chosen) >= 6:
            break
    extras = []
    for c in plan["allCameras"]:
        if c["index"] <= 100:
            continue
        cam = factory.get(c["name"])
        uv = cam.world_to_pixel(centre[:, None]).ravel()
        iw, ih = cam.im_size
        if 30 < uv[0] < iw - 30 and 30 < uv[1] < ih - 30:
            extras.append((np.linalg.norm(np.asarray(cam.pos[:2]) - centre[:2]), c))
    for _, c in sorted(extras, key=lambda a: a[0]):
        if all(abs(c["index"] - q["index"]) >= 4 for q in chosen):
            chosen.append(c)
        if len(chosen) >= 9:
            break
    folder = out / f"segment-{segment:02d}"
    folder.mkdir()
    views = []
    kps = {}
    descs = {}
    maps = {}
    matrices = {}
    images = []
    for i, c in enumerate(chosen):
        name = c["name"]
        uv = factory.get(name).world_to_pixel(P.reshape(3, -1)).reshape(2, h, w).astype(np.float32)
        with rio.open(project / "opensfm/undistorted/images" / name) as src:
            raw = src.read([1, 2, 3]).transpose(1, 2, 0)
            rawvalid = src.dataset_mask() > 0
        im = cv2.remap(raw, uv[0], uv[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = bilinear_support(rawvalid, uv)
        guard, boxes = dynamic_guard(im)
        f = im.astype(np.float32)
        gray = (
            (f.mean(-1) > 35)
            & (f.mean(-1) < 240)
            & ((f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1) < 0.30)
        )
        static = measurement_road & valid & ~guard & gray
        static = cv2.erode(static.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        kp, desc = sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (static * 255).astype(np.uint8)
        )
        kps[name] = kp
        descs[name] = desc
        maps[name] = uv
        matrices[name] = matrix(name)
        images.append(im)
        np.savez_compressed(
            folder / f"view-{i:02d}.npz",
            image=im,
            valid=valid,
            guard=guard,
            static=static,
            uv=uv,
            road=road,
            north=north,
        )
        Image.fromarray(im).save(folder / f"view-{i:02d}.jpg", quality=96)
        views.append(
            dict(
                name=name,
                index=c["index"],
                featureCount=len(kp),
                boxes=boxes,
                cachePath=str(folder / f"view-{i:02d}.npz"),
            )
        )
        print(segment, name, len(kp), len(boxes), flush=True)
    points = []
    rejected = 0
    # Use several anchors so measurements span both road edges, not one lane.
    for anchor in list(dict.fromkeys(v["name"] for v in (views[:3] + views[-2:]))):
        a = descs[anchor]
        if a is None or len(a) < 2:
            continue
        tracks = {i: [(anchor, np.array(k.pt))] for i, k in enumerate(kps[anchor])}
        for view in views:
            name = view["name"]
            b = descs[name]
            if name == anchor or b is None or len(b) < 2:
                continue
            ab = bf.knnMatch(a, b, k=2)
            ba = bf.knnMatch(b, a, k=2)
            rev = {
                m.queryIdx: m.trainIdx
                for pair in ba
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.72 * n.distance
            }
            for pair in ab:
                if len(pair) != 2:
                    continue
                m, n = pair
                if (
                    m.distance < 0.72 * n.distance
                    and rev.get(m.trainIdx) == m.queryIdx
                    and np.linalg.norm(
                        np.array(kps[name][m.trainIdx].pt) - np.array(kps[anchor][m.queryIdx].pt)
                    )
                    < 65
                ):
                    tracks[m.queryIdx].append((name, np.array(kps[name][m.trainIdx].pt)))
        for idx, items in tracks.items():
            if len(items) < 3:
                continue
            obs = []
            A = []
            for name, p in items:
                uv = np.array(
                    [
                        cv2.remap(
                            maps[name][k],
                            np.array([[p[0]]], np.float32),
                            np.array([[p[1]]], np.float32),
                            cv2.INTER_LINEAR,
                        )[0, 0]
                        for k in range(2)
                    ]
                )
                obs.append((name, uv))
                M = matrices[name]
                A.extend([uv[0] * M[2] - M[0], uv[1] * M[2] - M[1]])
            _, _, vt = np.linalg.svd(A)
            v = vt[-1]
            if abs(v[3]) < 1e-10:
                continue
            xyz = v[:3] / v[3]
            errors = []
            depths = []
            for name, p in obs:
                q = matrices[name] @ np.r_[xyz, 1.0]
                depths.append(q[2])
                errors.append(float(np.linalg.norm(q[:2] / q[2] - p)))
            if min(depths) <= 0 or max(errors) > 2.5 or np.percentile(errors, 90) > 2.0:
                rejected += 1
                continue
            world = xyz + offset
            gx, gy = (~tr0) * (world[0], world[1])
            lx = int(gx / 4)
            ly = int(gy / 4)
            if (
                lx < 0
                or lx >= roadlow.shape[1]
                or ly < 0
                or ly >= roadlow.shape[0]
                or not (maskdata["north"][ly, lx] or maskdata["south"][ly, lx])
            ):
                continue
            side = "north" if ly < mid[lx] else "south"
            points.append(
                dict(
                    track_id=f"seg{segment:02d}-{anchor}-{idx}",
                    xyz=world.tolist(),
                    origin="sfm-multiview-triangulation",
                    side=side,
                    observations=[
                        dict(
                            image_id=name,
                            pixel_xy=p.tolist(),
                            reprojection_error_px=e,
                            road_support="candidate",
                            dynamic=False,
                            evidence_id="asphalt-photo-domain-with-vehicle-box-exclusion",
                        )
                        for (name, p), e in zip(obs, errors)
                    ],
                )
            )
    fits = {}
    for side in ["north", "south"]:
        pp = [p for p in points if p["side"] == side]
        fit = fit_road_surface(
            pp,
            projectors,
            provenance=provenance,
            config=RoadSurfaceConfig(
                max_holdout_reprojection_p90_px=2.5,
                max_holdout_reprojection_px=5.0,
                max_original_reprojection_p90_px=2.5,
            ),
        )
        fits[side] = fit.report
        print(
            segment,
            side,
            len(pp),
            fit.report["geometryAccepted"],
            fit.report["reasons"],
            fit.report.get("holdoutHeightP90M"),
            flush=True,
        )
    rr = dict(
        segment=segment,
        window=[x0, y0, x1 - x0, y1 - y0],
        scale=2,
        views=views,
        points=points,
        triangulationRejected=rejected,
        fits=fits,
        seconds=time.time() - start,
    )
    (folder / "report.json").write_text(json.dumps(rr, indent=2))
    reports.append(rr)
    (out / "report.json").write_text(
        json.dumps(dict(state="running", provenance=provenance, segments=reports), indent=2)
    )
(out / "report.json").write_text(
    json.dumps(dict(state="completed", provenance=provenance, segments=reports), indent=2)
)
