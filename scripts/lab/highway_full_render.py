"""Render the declared whole highway, preserving its original world grid.

Diagnostic export is explicit: geometric failures/extrapolation are retained
in reports and cannot be silently promoted to a production acceptance.
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import Config, Detector, cameras_from_checkpoint
from model_engine.ogs2_corridor_projection import bilinear_support
from model_engine.ogs2_highway_background import HighwayBackgroundConfig, select_background
from model_engine.ogs2_highway_join import (
    cut_edges,
    preserve_boundary_foreground,
    sample_cached_body,
    select_highway_join,
)
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds

p = argparse.ArgumentParser()
p.add_argument("--scale", type=int, choices=[1, 2], default=2)
p.add_argument("--run-id", default="01")
p.add_argument("--body-cache", type=Path)
p.add_argument("--repair-west-from", type=Path)
p.add_argument("--finalize-from", type=Path)
p.add_argument("--diagnostic", action="store_true")
a = p.parse_args()
cv2.setNumThreads(4)
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,48}", a.run_id):
    raise ValueError("Run ID must be a short lowercase name")
r = Path("/volume/wzhang/mesh-research-workspace")
job = r / "runs/mangrove-077-20261013"
base = job / "highway-full-20261014-01"
project = job / "datasets/project"
out = base / f"full-render-scale{a.scale}-{a.run_id}"
out.mkdir(exist_ok=False)
file = base / "geometry-04/report.json"
while not file.exists() or json.loads(file.read_text())["state"] != "completed":
    time.sleep(3)
records = json.loads(file.read_text())["segments"]
plan = json.loads((base / "camera-corridor-plan.json").read_text())
tr0 = Affine(*plan["transform"])
mask = np.load(base / "road-mask-08/road-mask.npz")
loww = mask["road"].shape[1]
coords = np.arange(loww)
m = mask["median"]
top = mask["top"]
bottom = mask["bottom"]
models = {}
model_records = {}
failed = []
# The initially permissive southern search band included the parallel green
# barrier on the west. It is not the main carriageway. Keep the fitted height
# parameterisation unchanged, but exclude that extra search band from the
# actual replacement/seam domain. This is a declared scene scope constraint,
# not a claim that grey pixels automatically constitute a road.
effective_bottom = np.minimum(bottom, m + 160)
render_road = mask["road"] & (np.arange(mask["road"].shape[0])[:, None] <= effective_bottom)
body_out = out / "source-body-masks"
body_out.mkdir()
detector = (
    None if a.body_cache or a.finalize_from else Detector(r / "runtime/vehicle-models", "cuda:0", Config())
)


def observed_body(im):
    inputs = detector.processor(
        images=Image.fromarray(im), text="car. truck. bus. van. construction vehicle.", return_tensors="pt"
    ).to(detector.device)
    with detector.torch.inference_mode():
        outputs = detector.model(**inputs)
    result = detector.processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.20, text_threshold=0.18, target_sizes=[im.shape[:2]]
    )[0]
    boxes = result["boxes"].cpu().numpy()
    guard = np.zeros(im.shape[:2], bool)
    details = []
    if len(boxes):
        with detector.torch.inference_mode():
            detector.sam.set_image(im)
            bodies, quality, _ = detector.sam.predict(box=boxes, multimask_output=False)
        for box, body, q in zip(boxes, np.asarray(bodies).reshape(-1, *im.shape[:2]), np.ravel(quality)):
            left_col, t, rr, b = np.rint(box).astype(int)
            left_col = max(0, left_col)
            t = max(0, t)
            rr = min(im.shape[1], rr + 1)
            b = min(im.shape[0], b + 1)
            rect = np.zeros(guard.shape, bool)
            rect[t:b, left_col:rr] = True
            body = (body > 0) & rect
            area = float(body.sum() * (a.scale * 0.03) ** 2)
            unknown = bool(q < 0.8 or area < 0.3 or area > 90 or body.sum() < 0.2 * rect.sum())
            guard |= rect if unknown else body
            details.append(
                dict(box=box.tolist(), quality=float(q), areaM2=area, uncertainBoxRetained=unknown)
            )
    k = max(1, int(0.18 / (a.scale * 0.03))) * 2 + 1
    guard = cv2.dilate(guard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return guard, details


for side in ["north", "south"]:
    values = []
    for rec in records:
        f = rec["fits"][side]
        if not f["geometryAccepted"]:
            failed.append(dict(segment=rec["segment"], side=side, reasons=f["reasons"]))
        coef = f.get("coefficients")
        origin = f.get("originXY")
        if (
            coef is None
            or origin is None
            or np.linalg.norm(coef[:2]) > 0.2
            or f.get("fitInlierFraction", 0) < 0.65
        ):
            continue
        x = rec["window"][0] + rec["window"][2] / 2
        lx = int(x / 4)
        cy = (top[lx] + m[lx]) / 2 if side == "north" else (bottom[lx] + m[lx]) / 2
        X, Y = tr0 * (x, cy * 4)
        z = coef[0] * (X - origin[0]) + coef[1] * (Y - origin[1]) + coef[2]
        values.append(
            dict(
                x=x,
                z=z,
                bank=coef[1],
                segment=rec["segment"],
                geometryAccepted=f["geometryAccepted"],
                fitInlierFraction=f["fitInlierFraction"],
            )
        )
    if not values:
        raise RuntimeError("No train-supported road model for " + side)
    values = sorted(values, key=lambda q: q["x"])
    models[side] = values
    model_records[side] = values
if failed and not a.diagnostic:
    raise RuntimeError("Geometric gates failed; explicit diagnostic export required")


def road_height(gx, gy):
    lx = np.clip(gx / 4, 0, loww - 1)
    median = np.interp(lx, coords, m)
    v = {}
    for side in ["north", "south"]:
        entries = models[side]
        centrey = (
            (np.interp(lx, coords, top) + median) / 2
            if side == "north"
            else (np.interp(lx, coords, bottom) + median) / 2
        )
        z = np.interp(gx, [e["x"] for e in entries], [e["z"] for e in entries])
        bank = np.interp(gx, [e["x"] for e in entries], [e["bank"] for e in entries])
        v[side] = z + bank * ((gy - centrey * 4) * tr0.e)
    blend = np.clip((gy / 4 - median + 13) / 26, 0, 1)
    return (1 - blend) * v["north"] + blend * v["south"]


with rio.open(job / "vehicle-ogs2/ogs2_orthophoto.tif") as original:
    factory, data = cameras_from_checkpoint(project, original.crs)
    profile = original.profile.copy()
s = a.scale
width = int(np.ceil(24945 / a.scale))
height = int(np.ceil(4500 / a.scale))
canvas = np.memmap(out / "road-rgb.bin", mode="w+", dtype=np.uint8, shape=(height, width, 3))
source = np.memmap(out / "source-id.bin", mode="w+", dtype=np.uint16, shape=(height, width))
valid = np.memmap(out / "valid.bin", mode="w+", dtype=np.uint8, shape=(height, width))
retained = np.memmap(out / "retained.bin", mode="w+", dtype=np.uint8, shape=(height, width))
canvas[:] = 0
source[:] = 0
valid[:] = 0
retained[:] = 0
reports = []
body_canvas = np.memmap(out / "body.bin", mode="w+", dtype=np.uint8, shape=(height, width))
body_canvas[:] = 0
repair_join = None
if a.finalize_from:
    for target, name, dtype, shape in [
        (canvas, "road-rgb.bin", np.uint8, (height, width, 3)),
        (source, "source-id.bin", np.uint16, (height, width)),
        (valid, "valid.bin", np.uint8, (height, width)),
        (retained, "retained.bin", np.uint8, (height, width)),
        (body_canvas, "body.bin", np.uint8, (height, width)),
    ]:
        target[:] = np.memmap(a.finalize_from / name, mode="r", dtype=dtype, shape=shape)
    prior = json.loads((a.repair_west_from / "report.json").read_text())
    reports = json.loads((a.finalize_from / "progress.json").read_text())["segments"] + prior["segments"][3:]
    repair_join = json.loads((a.finalize_from / "repair-join.json").read_text())
    # Reconstruct conservative selected-body support in the unchanged eastern
    # mosaic from the original per-source masks; an absent body cache must not
    # be treated as evidence that a road is empty.
    body_canvas[:] = 0
    for rec in records:
        for vi, view in enumerate(rec["views"]):
            cache = np.load(
                a.repair_west_from / "source-body-masks" / f"segment-{rec['segment']:02d}-view-{vi:02d}.npz"
            )
            assert int(cache["scale"]) == 1
            bx = int(cache["x0"])
            by = int(cache["y0"])
            gh, gw = cache["guard"].shape
            measured = np.load(
                base
                / "full-render-scale2-06/source-body-masks"
                / f"segment-{rec['segment']:02d}-view-{vi:02d}.npz"
            )
            sy, sx = np.mgrid[:gh, :gw]
            uv = np.stack(
                [
                    (bx + sx + 0.5 - measured["x0"]) / measured["scale"] - 0.5,
                    (by + sy + 0.5 - measured["y0"]) / measured["scale"] - 0.5,
                ]
            ).astype(np.float32)
            measured_body, unknown = sample_cached_body(measured["guard"], uv)
            selected = (source[by : by + gh, bx : bx + gw] == view["index"]) & (
                valid[by : by + gh, bx : bx + gw] > 0
            )
            body_canvas[by : by + gh, bx : bx + gw] |= (measured_body & selected).astype(np.uint8)


def vertical_seam(A, B, ok, protected):
    return select_highway_join(A, B, ok, protected)


for rec in [] if a.finalize_from else records[:3] if a.repair_west_from else records:
    segment = rec["segment"]
    started = time.time()
    x0, y0, ww, hh = rec["window"]
    x1 = min(24945, x0 + ww + 400)
    y1 = min(4500, y0 + hh + 150)
    x0 = max(0, x0 - 400)
    y0 = max(0, y0 - 150)
    s = a.scale
    x0 = (x0 // s) * s
    y0 = (y0 // s) * s
    ww = int(np.ceil(x1 / s)) * s - x0
    hh = int(np.ceil(y1 / s)) * s - y0
    cw = ww // s
    ch = hh // s
    yy, xx = np.mgrid[:ch, :cw]
    gx = x0 + (xx + 0.5) * s
    gy = y0 + (yy + 0.5) * s
    X, Y = tr0 * (gx, gy)
    RZ = road_height(gx, gy)
    lx = np.clip((gx / 4).astype(int), 0, loww - 1)
    ly = np.clip((gy / 4).astype(int), 0, mask["road"].shape[0] - 1)
    road = render_road[ly, lx]
    north = gy / 4 < m[lx]
    tiletr = tr0 * Affine.translation(x0, y0) * Affine.scale(s)
    tilebounds = rio.transform.array_bounds(ch, cw, tiletr)
    with rio.open(job / "ogs2-surface.tif") as dsm:
        D = dsm.read(
            1,
            window=from_bounds(*tilebounds, dsm.transform),
            out_shape=(ch, cw),
            boundless=True,
            fill_value=dsm.nodata,
            resampling=rio.enums.Resampling.bilinear,
        )
        dvalid = np.isfinite(D) & (D != dsm.nodata)
    # Preserve the existing surface outside the ground road. A transition collar
    # changes surface heights, never blends photograph RGB or car silhouettes.
    distance = cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, 5) * s * 0.03
    weight = np.clip(1 - distance / 2.5, 0, 1)
    weight[road] = 1.0
    weight[(~road) & dvalid & (D > RZ + 1.0)] = 0.0
    Z = np.where(dvalid, weight * RZ + (1 - weight) * D, RZ)
    world = np.stack([X, Y, Z])
    images = []
    valids = []
    guards = []
    body_guards = []
    views = []
    alignment = []
    for view_index, view in enumerate(rec["views"]):
        name = view["name"]
        old = np.load(view["cachePath"])
        uv = factory.get(name).world_to_pixel(world.reshape(3, -1)).reshape(2, ch, cw).astype(np.float32)
        corrections = {}
        fids = {p["trackId"] for side in ["north", "south"] for p in rec["fits"][side]["fit"]}
        for side in ["north", "south"]:
            delta = []
            for point in rec["points"]:
                if point["side"] != side or point["track_id"] not in fids:
                    continue
                obs = next((o for o in point["observations"] if o["image_id"] == name), None)
                if obs is None:
                    continue
                px, py = (~tr0) * (point["xyz"][0], point["xyz"][1])
                z0 = float(road_height(np.array(px), np.array(py)))
                q = (
                    factory.get(name)
                    .world_to_pixel(np.array([[point["xyz"][0]], [point["xyz"][1]], [z0]]))
                    .ravel()
                )
                delta.append(np.array(obs["pixel_xy"]) - q)
            med = np.median(delta, axis=0) if len(delta) >= 12 else np.zeros(2)
            accepted = bool(len(delta) >= 12 and np.max(abs(med)) <= 6.0)
            corrections[side] = med if accepted else np.zeros(2)
            alignment.append(
                dict(
                    segment=segment,
                    image=name,
                    side=side,
                    measurements=len(delta),
                    offsetOriginalPixels=med.tolist(),
                    applied=accepted,
                    role=(
                        "bounded train-only camera-pixel residual translation; no independ"
                        "ent absolute control"
                    ),
                )
            )
        for side in ["north", "south"]:
            region = road & (north if side == "north" else ~north)
            for b in range(2):
                uv[b][region] += float(corrections[side][b])
        with rio.open(project / "opensfm/undistorted/images" / name) as photo:
            raw = photo.read([1, 2, 3]).transpose(1, 2, 0)
            rv = photo.dataset_mask() > 0
        im = cv2.remap(raw, uv[0], uv[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        v = bilinear_support(rv, uv)
        oh, ow = old["guard"].shape
        corners = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
        photo_corners = np.float32(
            [old["uv"][:, 0, 0], old["uv"][:, 0, -1], old["uv"][:, -1, -1], old["uv"][:, -1, 0]]
        )
        H = cv2.getPerspectiveTransform(corners, photo_corners)
        oldxy = (
            cv2.perspectiveTransform(uv.transpose(1, 2, 0).reshape(-1, 1, 2), np.linalg.inv(H))
            .reshape(ch, cw, 2)
            .astype(np.float32)
        )
        body_name = f"segment-{segment:02d}-view-{view_index:02d}.npz"
        if a.body_cache:
            cached = np.load(a.body_cache / body_name)
            header = json.loads((a.body_cache / body_name.replace(".npz", ".json")).read_text())
            if header["image"] != name:
                raise ValueError("Body-mask cache photograph identity mismatch")
            bx = (gx - cached["x0"]) / cached["scale"] - 0.5
            by = (gy - cached["y0"]) / cached["scale"] - 0.5
            body, unknown = sample_cached_body(cached["guard"], np.stack([bx, by]).astype(np.float32))
            g = body | unknown
            details = []
        else:
            g, details = observed_body(im)
            body = g & v
            unknown = ~v
        np.savez_compressed(body_out / body_name, guard=g, body=body, unknown=unknown, x0=x0, y0=y0, scale=s)
        (body_out / body_name.replace(".npz", ".json")).write_text(
            json.dumps(
                dict(
                    image=name,
                    window=[x0, y0, ww, hh],
                    scale=s,
                    objects=details,
                    role=(
                        "body and absent mask context stored separately; uncertain detecti"
                        "ons retain whole boxes"
                    ),
                ),
                indent=2,
            )
        )
        images.append(im)
        valids.append(v)
        guards.append(g)
        body_guards.append(body)
        views.append(view)
    result = select_background(
        np.stack(images),
        np.stack(valids),
        np.stack(guards),
        road,
        config=HighwayBackgroundConfig(
            shadow_padding_pixels=max(1, int(0.3 / (0.03 * s))),
            patch_collar_pixels=max(20, int(2 / (0.03 * s))),
        ),
    )
    # Store exact photograph identifiers, not tile-local donor numbers.
    ids = np.array([v["index"] for v in views], np.uint16)[result["source"]]
    outvalid = result["valid"]
    bg = result["image"]
    selected_body = np.take_along_axis(np.stack(body_guards), result["source"][None], axis=0)[0]
    remaining = result["remaining"] & selected_body
    # A continuous marking displaced by uncertain registration must not be
    # mistaken for a row of vehicle bodies that blocks every possible seam.
    # Compact unresolved temporal objects remain protected; all rejected
    # temporal proposals stay in the background report rather than disappearing.
    cc, lab, stats, _ = cv2.connectedComponentsWithStats(
        (result["remaining"] & ~selected_body).astype(np.uint8), 8
    )
    compact = (
        (stats[:, cv2.CC_STAT_AREA] * (s * 0.03) ** 2 < 150)
        & (stats[:, cv2.CC_STAT_WIDTH] < 6 * stats[:, cv2.CC_STAT_HEIGHT])
        & (stats[:, cv2.CC_STAT_HEIGHT] < 6 * stats[:, cv2.CC_STAT_WIDTH])
    )
    compact[0] = False
    remaining |= compact[lab]
    sx = x0 // s
    sy = y0 // s
    ex = sx + cw
    ey = sy + ch
    oldvalid = valid[sy:ey, sx:ex] > 0
    take = outvalid & ~oldvalid
    overlap = oldvalid & outvalid
    seam_report = dict(overlapPixels=int(overlap.sum()), pathAccepted=True)
    if overlap.any():
        columns = np.flatnonzero(overlap.any(0))
        left_col = int(columns.min())
        rr = int(columns.max()) + 1
        both = overlap[:, left_col:rr]
        left = canvas[sy:ey, sx + left_col : sx + rr].copy()
        right = bg[:, left_col:rr]
        protected = (retained[sy:ey, sx + left_col : sx + rr] > 0) | remaining[:, left_col:rr]
        # Outside actual common coverage a retained-source side is continued. A DP
        # path must cross the road only where both candidates are valid.
        road_local = road[:, left_col:rr]
        ok = both | ~road_local
        path = vertical_seam(left, right, ok, protected & road_local)
        if path is None:
            actual_body = (body_canvas[sy:ey, sx + left_col : sx + rr] > 0) | selected_body[:, left_col:rr]
            np.savez_compressed(
                out / f"join-{segment:02d}-diagnostic.npz",
                left=left,
                right=right,
                both=both,
                road=road_local,
                protected=protected,
                body=actual_body,
            )
            # Temporal RGB disagreement also flags displaced paint. Keep it as a
            # seam cost when its hard exclusion blocks the entire overlap; detected
            # bodies (including uncertain boxes) remain inviolable at both endpoints.
            path = select_highway_join(
                left,
                right,
                ok,
                actual_body & road_local,
                penalty=(protected & ~actual_body & road_local).astype(np.float32) * 40.0,
            )
            seam_report["temporalProtectionMode"] = (
                "soft RGB cost after infeasible hard temporal proposals; detected bodies remain hard"
            )
            if path is None:
                raise RuntimeError(
                    f"Segment {segment}: no foreground-safe seam; no midpoint fallback permitted"
                )
            seam_report["temporalCutEndpoints"] = int(
                (cut_edges(path) & protected & ~actual_body & road_local).sum()
            )
            protected = actual_body
        take[:, left_col:rr] |= path & both
        edge = cut_edges(path)
        check = edge & both & road_local
        diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean(-1)[check]
        seam_report.update(
            roadSeamPixels=len(diff),
            rgbP90=float(np.percentile(diff, 90)) if len(diff) else 0.0,
            rgbMax=float(diff.max()) if len(diff) else 0.0,
            protectedCutEndpoints=int((edge & protected & road_local).sum()),
        )
    canvas[sy:ey, sx:ex][take] = bg[take]
    source[sy:ey, sx:ex][take] = ids[take]
    valid[sy:ey, sx:ex][take] = 1
    retained[sy:ey, sx:ex][take] = remaining[take]
    body_canvas[sy:ey, sx:ex][take] = selected_body[take]
    rrr = dict(
        segment=segment,
        window=rec["window"],
        background=result["report"],
        alignment=alignment,
        join=seam_report,
        seconds=time.time() - started,
    )
    reports.append(rrr)
    Image.fromarray(bg).resize((max(1, cw // 2), max(1, ch // 2))).save(
        out / f"segment-{segment:02d}.jpg", quality=96
    )
    (out / f"segment-{segment:02d}.json").write_text(json.dumps(rrr, indent=2))
    (out / "progress.json").write_text(
        json.dumps(dict(state="running", completed=len(reports), segments=reports), indent=2)
    )
    print(
        segment,
        "patched",
        len(result["report"]["applied"]),
        "retained",
        result["report"]["unresolvedPixels"],
        "join",
        seam_report,
        "seconds",
        round(time.time() - started),
        flush=True,
    )
# A west repair reuses the finished native raster to the east; it does not
# recompute unchanged photographs or erase the original failed join evidence.
if a.repair_west_from and not a.finalize_from:
    while not (a.repair_west_from / "report.json").exists():
        time.sleep(3)
    prior = json.loads((a.repair_west_from / "report.json").read_text())
    if prior["scale"] != a.scale or prior["width"] != width or prior["height"] != height:
        raise ValueError("Repair raster grid mismatch")
    previous = {
        k: np.memmap(a.repair_west_from / fn, mode="r", dtype=dtype, shape=shape)
        for k, fn, dtype, shape in [
            ("image", "road-rgb.bin", np.uint8, (height, width, 3)),
            ("source", "source-id.bin", np.uint16, (height, width)),
            ("valid", "valid.bin", np.uint8, (height, width)),
            ("retained", "retained.bin", np.uint8, (height, width)),
        ]
    }
    left_col = 5000 // a.scale
    rr = 6600 // a.scale
    yy, xx = np.mgrid[:height, left_col:rr]
    lx = np.clip((xx * a.scale / 4).astype(int), 0, loww - 1)
    ly = np.clip((yy * a.scale / 4).astype(int), 0, mask["road"].shape[0] - 1)
    road_local = render_road[ly, lx]
    both = (valid[:, left_col:rr] > 0) & (previous["valid"][:, left_col:rr] > 0)
    protected = (retained[:, left_col:rr] > 0) | (previous["retained"][:, left_col:rr] > 0)
    path = vertical_seam(
        canvas[:, left_col:rr], previous["image"][:, left_col:rr], both | ~road_local, protected & road_local
    )
    if path is None:
        raise RuntimeError("West repair cannot safely join the unchanged eastern raster")
    edges = cut_edges(path)
    diff = np.abs(
        canvas[:, left_col:rr].astype(np.float32) - previous["image"][:, left_col:rr].astype(np.float32)
    ).mean(-1)[edges & both & road_local]
    repair_join = dict(
        pathAccepted=True,
        overlapFullPixels=[5000, 6600],
        protectedCutEndpoints=int((edges & protected & road_local).sum()),
        rgbP90=float(np.percentile(diff, 90)) if len(diff) else 0.0,
        rgbMax=float(diff.max()) if len(diff) else 0.0,
    )
    take_previous = (valid == 0) & (previous["valid"] > 0)
    take_previous[:, rr:] = previous["valid"][:, rr:] > 0
    take_previous[:, left_col:rr] |= path & both
    for key, target in [("image", canvas), ("source", source), ("valid", valid), ("retained", retained)]:
        target[take_previous] = previous[key][take_previous]
    reports += prior["segments"][3:]
    (out / "repair-join.json").write_text(json.dumps(repair_join, indent=2))
    print("REPAIR_JOIN", repair_join, flush=True)
# Close the main-road replacement in a collar matching the original context.
tr = tr0 * Affine.scale(s)
yy, xx = np.mgrid[:height, :width]
lx = np.clip((xx * s / 4).astype(int), 0, loww - 1)
ly = np.clip((yy * s / 4).astype(int), 0, mask["road"].shape[0] - 1)
road = render_road[ly, lx]
with rio.open(job / "vehicle-ogs2/ogs2_orthophoto.tif") as original:
    old = original.read(
        [1, 2, 3],
        window=Window(0, 17000, 24945, 4500),
        out_shape=(3, height, width),
        resampling=rio.enums.Resampling.bilinear,
    ).transpose(1, 2, 0)
    ov = original.dataset_mask(window=Window(0, 17000, 24945, 4500), out_shape=(height, width)) > 0
    rgbdiff = np.abs(old.astype(np.float32) - canvas.astype(np.float32)).mean(-1)
    paths = {}
    boundary_qc = {}
    for side in ["top", "bottom"]:
        centre = (
            np.interp(np.arange(width) * s / 4, coords, top if side == "top" else effective_bottom) * 4 / s
        )
        radius = max(10, int(4 / 0.03 / s))
        lower = -radius if side == "top" else -int(1 / 0.03 / s)
        upper = int(1 / 0.03 / s) if side == "top" else radius
        band = (yy >= centre[None] + lower) & (yy <= centre[None] + upper)
        observed = (band & (valid > 0)).any(0)
        available = band.copy()
        # Use the exact cut-edge-aware join on the world grid. The old slope-one
        # local-offset DP could become infeasible when a curb/guard shifts, and
        # following its all-infinite backpointers caused negative array indices.
        protected = (retained > 0) & band
        protected[:, ~observed] = False
        boundary_image = np.where((valid > 0)[..., None], np.asarray(canvas), old)
        cut = select_highway_join(
            old.transpose(1, 0, 2), boundary_image.transpose(1, 0, 2), available.T, protected.T
        )
        mode = "hard retained foreground"
        if cut is None:
            detected = (body_canvas > 0) & band
            detected[:, ~observed] = False
            cut = select_highway_join(
                old.transpose(1, 0, 2),
                boundary_image.transpose(1, 0, 2),
                available.T,
                detected.T,
                penalty=(protected & ~detected).T.astype(np.float32) * 40.0,
            )
            mode = "hard detected body with soft temporal/uncertain-context cost"
        if cut is None:
            cut = select_highway_join(
                old.transpose(1, 0, 2),
                boundary_image.transpose(1, 0, 2),
                available.T,
                np.zeros_like(available.T),
                penalty=protected.T.astype(np.float32) * 40.0,
            )
            mode = (
                "whole foreground enclosure after infeasible continuous exterior s"
                "eam; see boundary closure report"
            )
        if cut is None:
            raise RuntimeError("No exterior boundary in the declared world-grid band")
        line = cut.argmax(1)
        paths[side] = line
        values = rgbdiff[line, np.arange(width)][observed]
        edges = cut_edges(cut).T
        boundary_qc[side] = dict(
            pathAccepted=True,
            protectionMode=mode,
            observedColumns=int(observed.sum()),
            unobservedColumnsKeptOriginal=int((~observed).sum()),
            rgbP90=float(np.percentile(values, 90)) if len(values) else None,
            rgbMax=float(values.max()) if len(values) else None,
            detectedBodyCutEndpoints=int((edges & (body_canvas > 0) & band).sum()),
        )
    replacement = (yy >= paths["top"][None]) & (yy <= paths["bottom"][None]) & (valid > 0)
    permitted = road | (cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, 5) * s * 0.03 <= 4.5)
    replacement, boundary_qc["wholeForegroundClosure"] = preserve_boundary_foreground(
        replacement, body_canvas > 0, valid > 0, road, permitted, padding_pixels=max(1, int(0.3 / 0.03 / s))
    )
    new = old.copy()
    new[replacement] = canvas[replacement]
    prov = np.where(replacement, source, 0).astype(np.uint16)
    uncovered = int((road & (valid == 0)).sum())
    core_coverage = float((road & replacement).sum() / max(1, road.sum()))
    profile.update(
        width=width,
        height=height,
        transform=tr,
        count=4,
        dtype="uint8",
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    alpha = np.where(ov | replacement, 255, 0).astype(np.uint8)
    with rio.open(out / "CANDIDATE-full-highway-corridor.tif", "w", **profile) as dst:
        dst.write(np.concatenate([new, alpha[..., None]], -1).transpose(2, 0, 1))
        dst.update_tags(
            ACCEPTED="false",
            STATUS="diagnostic whole-highway candidate",
            ORIGINAL_REFERENCE=str(job / "vehicle-ogs2/ogs2_orthophoto.tif"),
        )
    one = profile.copy()
    one.update(count=1, dtype="uint16", nodata=0)
    with rio.open(out / "source-photo-provenance.tif", "w", **one) as dst:
        dst.write(prov, 1)
    one.update(dtype="uint8")
    with rio.open(out / "retained-foreground.tif", "w", **one) as dst:
        dst.write(np.where(replacement, retained, 0).astype(np.uint8), 1)
    np.savez_compressed(
        out / "replacement-scope.npz",
        replacement=replacement,
        road=road,
        top=paths["top"],
        bottom=paths["bottom"],
    )
    Image.fromarray(new).resize((6237, 1125)).save(out / "after-highway-overview.jpg", quality=98)
    Image.fromarray(old).resize((6237, 1125)).save(out / "before-highway-overview.jpg", quality=98)
    cmp = Image.new("RGB", (6237, 2250), "white")
    cmp.paste(Image.open(out / "before-highway-overview.jpg"), (0, 0))
    cmp.paste(Image.open(out / "after-highway-overview.jpg"), (0, 1125))
    cmp.save(out / "before-after-highway.jpg", quality=98)
    report = dict(
        state="completed",
        accepted=False,
        scope="entire declared visible main highway, both carriageways; scene-specific reviewed scope seeds",
        scale=s,
        crs=profile["crs"].to_string(),
        transform=list(tr)[:6],
        bounds=list(rio.transform.array_bounds(height, width, tr)),
        width=width,
        height=height,
        roadCoreCoverage=core_coverage,
        uncoveredRoadPixels=uncovered,
        replacedPixels=int(replacement.sum()),
        remainingForegroundPixels=int((replacement & (retained > 0)).sum()),
        geometryFailedSegments=failed,
        models=model_records,
        modelUsesHoldoutForSelection=False,
        boundary=boundary_qc,
        segments=reports,
        repairFrom=str(a.repair_west_from) if a.repair_west_from else None,
        repairJoin=repair_join,
        limitations=[
            (
                "Automatic road proposals failed and were not adopted; scope seeds"
                " were visually reviewed on real source photos."
            ),
            (
                "Road heights outside training support and failed held-out geometr"
                "y remain diagnostic; stored coordinates are not an independent ac"
                "curacy certificate."
            ),
            (
                "Dynamic masks may miss objects; source IDs, unresolved masks and "
                "local visual checks remain necessary."
            ),
            "No generative inpainting or photograph alpha averaging.",
        ],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETED", out, core_coverage, uncovered, report["remainingForegroundPixels"], flush=True)
