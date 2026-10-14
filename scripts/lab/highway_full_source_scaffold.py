import json
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import cameras_from_checkpoint
from model_engine.ogs2_corridor_projection import bilinear_support
from PIL import Image
from rasterio.transform import Affine

r = Path("/volume/wzhang/mesh-research-workspace")
job = r / "runs/mangrove-077-20261013"
base = job / "highway-full-20261014-01"
out = base / "source-scaffold-02"
out.mkdir(exist_ok=False)
plan = json.loads((base / "camera-corridor-plan.json").read_text())
tr = Affine(*plan["transform"]) * Affine.scale(4)
w = 6237
h = 1125
image = np.zeros((h, w, 3), np.uint8)
ids = np.zeros((h, w), np.uint16)
records = []
with rio.open(job / "vehicle-ogs2/ogs2_orthophoto.tif") as ref:
    factory, data = cameras_from_checkpoint(job / "datasets/project", ref.crs)
cameras = [c for c in plan["allCameras"] if c["index"] <= 95]
for x0 in range(0, w, 350):
    x1 = min(w, x0 + 350)
    yy, xx = np.mgrid[:h, x0:x1]
    X, Y = tr * (xx + 0.5, yy + 0.5)
    Z = np.full(X.shape, 3.0)
    cy = 450
    cx = (x0 + x1) // 2
    centre = np.array([*tr * (cx + 0.5, cy + 0.5), 3.0])
    candidates = []
    for c in cameras:
        cam = factory.get(c["name"])
        uv = cam.world_to_pixel(centre[:, None]).ravel()
        iw, ih = cam.im_size
        if 1 < uv[0] < iw - 2 and 1 < uv[1] < ih - 2:
            candidates.append((np.linalg.norm(np.array(cam.pos[:2]) - centre[:2]), c))
    chosen = sorted(candidates, key=lambda a: a[0])[:4]
    for rank, (_, c) in enumerate(chosen):
        uv = (
            factory.get(c["name"])
            .world_to_pixel(np.stack([X, Y, Z]).reshape(3, -1))
            .reshape(2, h, x1 - x0)
            .astype(np.float32)
        )
        with rio.open(job / "datasets/project/opensfm/undistorted/images" / c["name"]) as src:
            raw = src.read([1, 2, 3]).transpose(1, 2, 0)
            valid = src.dataset_mask() > 0
        take = bilinear_support(valid, uv) & (ids[:, x0:x1] == 0)
        rgb = cv2.remap(raw, uv[0], uv[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        image[:, x0:x1][take] = rgb[take]
        ids[:, x0:x1][take] = c["index"]
    records.append(
        dict(
            x0=x0, x1=x1, photos=[c["name"] for _, c in chosen], coveredPixels=int((ids[:, x0:x1] > 0).sum())
        )
    )
    print(records[-1], flush=True)
Image.fromarray(image).save(out / "source-scaffold.jpg", quality=98)
np.savez_compressed(out / "source-scaffold.npz", rgb=image, source_id=ids)
(out / "report.json").write_text(
    json.dumps(
        dict(
            state="completed",
            purpose="raw-photo road search and static-feature measurement scaffold only",
            provisionalHeightM=3.0,
            heightCertified=False,
            notFinalOrthophoto=True,
            records=records,
        ),
        indent=2,
    )
)
