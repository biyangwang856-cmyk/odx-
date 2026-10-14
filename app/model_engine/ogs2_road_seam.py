"""Bounded, monotone hard seams between two calibrated real road projections.

Inputs are already on one physical grid. This module does not estimate geometry,
infer empty road, correct exposure, warp images, or certify a whole orthophoto.
The caller must supply independently verified static/geometry support and whole
vehicle plus measured-shadow masks for BOTH exposures. RGB similarity is only
an extra rejection test; it never establishes semantic or geometric truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class RoadSeamConfig:
    max_pixels: int = 2_000_000
    max_step: int = 2
    max_rgb_difference: float = 18.0
    max_gradient_difference: float = 12.0
    step_penalty: float = 2.0


@dataclass(frozen=True)
class RoadSeamResult:
    accepted: bool
    reason: str
    take_b: np.ndarray | None
    source_index: np.ndarray | None
    seam_mask: np.ndarray | None
    qc: dict
    bindings: dict


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask on the common grid")
    return a


def _images(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or b.dtype != np.uint8 or b.shape != a.shape:
        raise ValueError("Require two uint8 HxWx3 real projections on one grid")
    if min(a.shape[:2]) < 3:
        raise ValueError("Seam context requires at least three rows and columns")
    return a, b


def _prefix(values):
    return np.concatenate(([0], np.cumsum(values)))


def _boundary(take_b):
    seam = np.zeros(take_b.shape, bool)
    across = take_b[:, 1:] != take_b[:, :-1]
    seam[:, 1:] |= across
    seam[:, :-1] |= across
    across = take_b[1:] != take_b[:-1]
    seam[1:] |= across
    seam[:-1] |= across
    return seam


def select_road_seam(
    rgb_a, rgb_b, valid_a, valid_b, trusted_static, geometry_verified,
    forbidden_cut_a, forbidden_cut_b, hard_a, hard_b, *,
    orientation="vertical", markings=None, marking_continuity_verified=None,
    config=RoadSeamConfig(),
):
    """Select a complete monotone source boundary, or return an explicit refusal.

    ``vertical`` selects A left/B right; ``horizontal`` selects A above/B below.
    A boundary moves at most ``max_step`` pixels per row (column for horizontal).
    Hard A/B cores may fix source-side borders or a whole object. Every output
    pixel must have valid support in its selected source: there is no fallback.
    All actual cut-edge endpoints, including stairs of the path, must satisfy
    the supplied static/geometry evidence and photometric limits. Markings, when
    supplied, are forbidden to cross unless their continuity is explicitly
    verified by the caller. Omitting their mask is recorded, never certification.
    """
    a, b = _images(rgb_a, rgb_b)
    shape = a.shape[:2]
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be vertical or horizontal")
    if isinstance(config.max_pixels, bool) or not isinstance(config.max_pixels, int) or config.max_pixels < 9:
        raise ValueError("Invalid pixel budget")
    if a.shape[0] * a.shape[1] > config.max_pixels:
        raise ValueError("Seam exceeds bounded pixel budget")
    if isinstance(config.max_step, bool) or not isinstance(config.max_step, int) or not 0 <= config.max_step <= 4:
        raise ValueError("max_step must be an integer from zero to four")
    for value in (config.max_rgb_difference, config.max_gradient_difference, config.step_penalty):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("Photometric limits and step penalty must be finite and nonnegative")
    va, vb, static, geo, fa, fb, ha, hb = [
        _mask(x, shape, n) for x, n in zip(
            (valid_a, valid_b, trusted_static, geometry_verified, forbidden_cut_a, forbidden_cut_b, hard_a, hard_b),
            ("valid_a", "valid_b", "trusted_static", "geometry_verified", "forbidden_cut_a", "forbidden_cut_b", "hard_a", "hard_b"),
        )
    ]
    if markings is None and marking_continuity_verified is not None:
        raise ValueError("Marking continuity proof requires an explicit marking mask")
    paint = np.zeros(shape, bool) if markings is None else _mask(markings, shape, "markings")
    paint_ok = np.zeros(shape, bool) if marking_continuity_verified is None else _mask(marking_continuity_verified, shape, "marking_continuity_verified")
    qc = {"schema": "ogs2-road-hard-seam/v1", "orientation": orientation,
          "config": asdict(config),
          "wholeOrthophotoCertified": False, "callerStaticEvidenceRequired": True,
          "callerGeometryEvidenceRequired": True, "markingMaskProvided": markings is not None,
          "markingContinuityMaskProvided": marking_continuity_verified is not None,
          "rgbSimilarityUsedAsStaticProof": False, "rgbWarped": False,
          "rgbBlended": False, "rgbExposureAdjusted": False, "fallbackPixels": 0}
    bindings = {"sourceASha256": _digest(a), "sourceBSha256": _digest(b)}
    bindings["inputEvidenceSha256"] = {
        name: _digest(value) for name, value in (
            ("validA", va), ("validB", vb), ("trustedStatic", static),
            ("geometryVerified", geo), ("forbiddenCutA", fa), ("forbiddenCutB", fb),
            ("hardA", ha), ("hardB", hb), ("markings", paint), ("markingContinuity", paint_ok),
        )
    }

    def reject(reason):
        return RoadSeamResult(False, reason, None, None, None, {**qc, "accepted": False}, bindings)

    if (ha & hb).any():
        return reject("conflicting-hard-source-cores")
    if not ha.any() or not hb.any():
        return reject("both-explicit-source-cores-required")
    af, bf = a.astype(np.float32), b.astype(np.float32)
    mismatch = np.max(np.abs(af - bf), axis=2)
    ay, ax = np.gradient(af, axis=(0, 1))
    by, bx = np.gradient(bf, axis=(0, 1))
    gradient_difference = np.maximum(np.max(np.abs(ax - bx), axis=2), np.max(np.abs(ay - by), axis=2))
    admissible = va & vb & static & geo & ~fa & ~fb & (~paint | paint_ok)
    admissible &= (mismatch <= config.max_rgb_difference) & (gradient_difference <= config.max_gradient_difference)
    pixel_cost = 1.0 + mismatch.astype(np.float64) + gradient_difference.astype(np.float64)
    transpose = orientation == "horizontal"
    va, vb, ha, hb, allowed, cost = [x.T if transpose else x for x in (va, vb, ha, hb, admissible, pixel_cost)]
    h, w = va.shape
    splits = np.arange(1, w)
    row_ok = np.empty((h, w - 1), bool)
    for y in range(h):
        pa, pb, pha, phb = [_prefix(x) for x in (~va[y], ~vb[y], ha[y], hb[y])]
        row_ok[y] = (pa[splits] == 0) & (pb[-1] - pb[splits] == 0)
        row_ok[y] &= (phb[splits] == 0) & (pha[-1] - pha[splits] == 0)
    row_ok &= allowed[:, :-1] & allowed[:, 1:]
    if not row_ok.any(axis=1).all():
        return reject("no-valid-static-split-in-some-road-row")
    horizontal_cost = cost[:, :-1] + cost[:, 1:]
    parents = np.full((h, w - 1), -1, np.int32)
    previous = np.where(row_ok[0], horizontal_cost[0], np.inf)
    # Each transition tests every vertical cut edge made by a staircase. Only
    # checking the seam centre pixel would let turns cross a vehicle or no-data.
    for y in range(1, h):
        bad = _prefix(~(allowed[y - 1] & allowed[y]))
        vertical_cost = _prefix(cost[y - 1] + cost[y])
        current = np.full(w - 1, np.inf)
        for delta in range(-config.max_step, config.max_step + 1):
            start, stop = max(0, delta), min(w - 1, w - 1 + delta)
            if stop <= start:
                continue
            dst = np.arange(start, stop)
            src = dst - delta
            left, right = np.minimum(dst + 1, src + 1), np.maximum(dst + 1, src + 1)
            legal = (bad[right] - bad[left] == 0) & row_ok[y, dst]
            candidate = previous[src] + vertical_cost[right] - vertical_cost[left] + config.step_penalty * abs(delta) + horizontal_cost[y, dst]
            improve = legal & (candidate < current[dst])
            current[dst[improve]] = candidate[improve]
            parents[y, dst[improve]] = src[improve]
        previous = current
    if not np.isfinite(previous).any():
        return reject("no-connected-verified-static-seam")
    index = int(np.argmin(previous))
    energy = float(previous[index])
    path = np.empty(h, np.int32)
    for y in range(h - 1, -1, -1):
        path[y] = index + 1
        if y:
            index = int(parents[y, index])
            if index < 0:
                raise RuntimeError("Broken dynamic-programming parent path")
    take_b = np.arange(w)[None, :] >= path[:, None]
    if (take_b & ha).any() or (~take_b & hb).any() or (~take_b & ~va).any() or (take_b & ~vb).any():
        raise RuntimeError("Hard seam violated a source/coverage constraint")
    take_b = take_b.T.copy() if transpose else take_b
    seam = _boundary(take_b)
    if not seam.any() or (seam & ~admissible).any():
        raise RuntimeError("An actual cut edge left verified static support")
    ids = np.where(take_b, 2, 1).astype(np.uint8)
    qc.update(accepted=True, seamPixels=int(seam.sum()), sourceAPixels=int((~take_b).sum()),
              sourceBPixels=int(take_b.sum()), seamMaxRgbDifference=float(mismatch[seam].max()),
              seamMaxGradientDifference=float(gradient_difference[seam].max()),
              energy=energy, path=path.tolist(), invalidSelectedPixels=0,
              forbiddenCutPixels=0, unsupportedCutPixels=0)
    bindings.update(takeBSha256=_digest(take_b), sourceIndexSha256=_digest(ids), seamSha256=_digest(seam))
    for x in (take_b, ids, seam):
        x.setflags(write=False)
    return RoadSeamResult(True, "verified-monotone-hard-seam", take_b, ids, seam, qc, bindings)


def compose_road_seam(rgb_a, rgb_b, result):
    """Copy one real source per pixel; reject altered inputs or an invalid plan."""
    a, b = _images(rgb_a, rgb_b)
    if not isinstance(result, RoadSeamResult) or not result.accepted:
        raise ValueError("Only an accepted road seam can be composed")
    for name, value in (("sourceASha256", a), ("sourceBSha256", b),
                        ("takeBSha256", result.take_b), ("sourceIndexSha256", result.source_index),
                        ("seamSha256", result.seam_mask)):
        if _digest(value) != result.bindings.get(name):
            raise ValueError("Source or accepted seam changed after selection")
    take_b = _mask(result.take_b, a.shape[:2], "accepted take_b")
    if not np.array_equal(result.source_index, np.where(take_b, 2, 1).astype(np.uint8)):
        raise ValueError("Source index disagrees with the accepted assignment")
    output = a.copy()
    output[take_b] = b[take_b]
    if not np.array_equal(output[~take_b], a[~take_b]) or not np.array_equal(output[take_b], b[take_b]):
        raise RuntimeError("Output failed exact real-source pixel equality")
    return output, result.source_index.copy(), {**result.qc, "mixedPixels": 0,
        "realSourcePixelEquality": True, "outputSha256": _digest(output)}
