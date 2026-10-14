import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import read_offset
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

r = Path("/volume/wzhang/mesh-research-workspace")
job = r / "runs/mangrove-077-20261013"
out = job / "highway-full-20261014-01"
out.mkdir(exist_ok=False)
repo = r / "mesh-research"
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
rec = json.loads((job / "datasets/project/opensfm/undistorted/reconstruction.json").read_text())[0]
offset = read_offset(job / "datasets/project")
cameras = []
with rio.open(job / "vehicle-ogs2/ogs2_orthophoto.tif") as src:
    win = Window(0, 17000, src.width, 4500)
    tr = src.window_transform(win)
    im = src.read(
        [1, 2, 3], window=win, out_shape=(3, 1125, (src.width + 3) // 4), resampling=Resampling.average
    ).transpose(1, 2, 0)
    Image.fromarray(im).save(out / "before-highway-overview.jpg", quality=95)
    # A broad route proposal from elongated asphalt-colour components. This
    # search does not assert that grey pixels or closed holes are road truth.
    f = im.astype(float)
    gray = (
        (f.mean(-1) > 55) & (f.mean(-1) < 235) & ((f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1) < 0.22)
    )
    gray = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 13), np.uint8))
    count, lab, stats, c = cv2.connectedComponentsWithStats(gray)
    chosen = sorted(range(1, count), key=lambda k: stats[k, cv2.CC_STAT_WIDTH], reverse=True)[:8]
    print("GREY_COMPONENTS", [(k, stats[k].tolist()) for k in chosen])
    annotated = im.copy()
    for k in chosen:
        x, y, w, h, area = stats[k]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 2)
    Image.fromarray(annotated).save(out / "road-component-proposal.jpg", quality=95)
    np.savez_compressed(out / "road-grey-proposals.npz", gray=gray, labels=lab, stats=stats)
    for name, shot in rec["shots"].items():
        R = cv2.Rodrigues(np.array(shot["rotation"]))[0]
        C = -R.T @ np.array(shot["translation"]) + offset
        col, row = (~src.transform) * (C[0], C[1])
        i = int(re.search(r"_(\d+)_V", name).group(1))
        cameras.append(
            dict(
                name=name,
                index=i,
                position=C.tolist(),
                grid=[float(col), float(row)],
                captureTime=shot.get("capture_time"),
            )
        )
    (out / "camera-corridor-plan.json").write_text(
        json.dumps(
            dict(
                head=head,
                window=list(win.flatten()),
                transform=list(tr)[:6],
                crs=str(src.crs),
                allCameras=sorted(cameras, key=lambda x: x["index"]),
                scope=(
                    "full lower main highway within photo-supported map extent; both c"
                    "arriageways; excludes other streets"
                ),
            ),
            indent=2,
        )
    )
    print(
        "EARLY_CAMERAS",
        [
            (c["index"], [round(a, 1) for a in c["grid"]], round(c["position"][2], 1))
            for c in cameras
            if c["index"] <= 100
        ],
    )
print(str(out))
