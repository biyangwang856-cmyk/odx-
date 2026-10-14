"""Scene-reviewed main highway scope; visual seeds are not ground control."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

r = Path("/volume/wzhang/mesh-research-workspace")
base = r / "runs/mangrove-077-20261013/highway-full-20261014-01"
out = base / "road-mask-08"
out.mkdir(exist_ok=False)
im = np.load(base / "source-scaffold-02/source-scaffold.npz")["rgb"]
f = im.astype(np.float32)
valid = im.max(-1) > 0
h, w = im.shape[:2]
xx = np.arange(w)
reviewed = np.array(
    [
        [400, 509],
        [1200, 389],
        [2400, 324],
        [3200, 357],
        [4000, 448],
        [4700, 535],
        [5300, 645],
        [5800, 715],
        [6200, 770],
    ],
    float,
)
coef = np.polyfit(reviewed[:, 0], reviewed[:, 1], 3)
smooth = np.polyval(coef, xx)
pink = (f[..., 0] > 75) & (f[..., 0] > f[..., 1] * 1.15) & (f[..., 0] > f[..., 2] * 1.05) & valid
white = (f.min(-1) > 155) & (f.max(-1) - f.min(-1) < 65) & valid
gray = (
    (f.mean(-1) > 40)
    & (f.mean(-1) < 225)
    & ((f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1) < 0.25)
    & valid
)
red = cv2.boxFilter(pink.astype(np.float32), -1, (101, 7))
paint = cv2.boxFilter(white.astype(np.float32), -1, (151, 9))
asphalt = cv2.boxFilter(gray.astype(np.float32), -1, (101, 19))
Y = np.arange(h)[:, None]


def path(field, centre, lo, hi):
    grid = np.arange(0, w, 12)
    ys = np.arange(h)
    cost = -field[:, grid].T.copy()
    back = np.zeros(cost.shape, np.int16)
    for i, x in enumerate(grid):
        cost[i, (ys < centre[x] + lo) | (ys > centre[x] + hi)] = np.inf
    delta = np.arange(-4, 5)
    best = cost[0]
    for i in range(1, len(grid)):
        values = np.full((len(delta), h), np.inf)
        for k, d in enumerate(delta):
            old = ys + d
            ok = (old >= 0) & (old < h)
            values[k, ok] = best[old[ok]] + 0.025 * abs(d)
        k = np.argmin(values, axis=0)
        back[i] = ys + delta[k]
        best = cost[i] + values[k, ys]
    yy = np.empty(len(grid), int)
    yy[-1] = np.argmin(best)
    for i in range(len(grid) - 1, 0, -1):
        yy[i - 1] = back[i, yy[i]]
    return gaussian_filter1d(np.interp(xx, grid, yy), 16)


median = path(12 * red + paint, smooth, -30, 30)


# Bright continuous curb, with asphalt inward and a lower asphalt density
# outward. Cars are too short longitudinally to dominate the 18m averaging.
def shifted(a, d):
    return cv2.remap(
        a,
        np.broadcast_to(xx[None], (h, w)).astype(np.float32),
        np.broadcast_to(np.arange(h)[:, None] + d, (h, w)).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


top = path(paint + 0.7 * shifted(asphalt, 20) - 0.6 * shifted(asphalt, -20), median, -185, -80)
bottom = path(paint + 0.7 * shifted(asphalt, -20) - 0.6 * shifted(asphalt, 20), median, 110, 235)
road = (Y >= top) & (Y <= bottom) & valid
north = (Y > top + 12) & (Y < median - 13) & valid
south = (Y > median + 13) & (Y < bottom - 12) & valid
overlay = im.copy()
overlay[north] = (overlay[north] * 0.75 + np.array([20, 240, 50]) * 0.25).astype(np.uint8)
overlay[south] = (overlay[south] * 0.75 + np.array([30, 120, 255]) * 0.25).astype(np.uint8)
for line, c in [(median, (255, 0, 255)), (top, (255, 255, 0)), (bottom, (255, 255, 0))]:
    cv2.polylines(overlay, [np.stack([xx, line], 1).astype(np.int32)], False, c, 2)
Image.fromarray(overlay).save(out / "road-mask-review.jpg", quality=96)
np.savez_compressed(
    out / "road-mask.npz", road=road, north=north, south=south, top=top, bottom=bottom, median=median
)
(out / "report.json").write_text(
    json.dumps(
        dict(
            role=(
                "smooth highway divider and continuous curbs from real photo appea"
                "rance; not ground-height certification"
            ),
            scale=4,
            roadPixels=int(road.sum()),
            dividerScopeSeeds=reviewed.tolist(),
            scopeSeedSource=(
                "visual review of real projected source photos; not ground control or height measurement"
            ),
            coefficient=coef.tolist(),
            routeSamples=[[x, float(median[x]), float(top[x]), float(bottom[x])] for x in range(400, w, 400)],
        ),
        indent=2,
    )
)
print((out / "report.json").read_text())
