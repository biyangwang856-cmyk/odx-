"""Additional unwarped static measurements for low-texture road markings.

GFTT supplies corner locations and symmetric local NCC supplies independently
measured correspondences. Search is wider than the acceptance tolerance; no
correspondence is selected because its measured displacement is small. The
fixed spatial holdout includes every quality-accepted match, including outliers.
This provider never changes an image, camera, height, coordinate or RGB pixel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import cv2
import numpy as np

from .ogs2_joint_geometry import GeometryConfig, independent_static_evidence


@dataclass(frozen=True)
class NCCConfig:
    patch_radius: int = 5
    search_radius: int = 8
    max_corners: int = 3000
    corner_quality: float = 0.005
    min_patch_std: float = 2.0
    min_correlation: float = 0.90
    min_peak_gap: float = 0.03
    peak_exclusion_radius: int = 1
    max_reverse_error: float = 0.5
    subpixel: bool = True
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


def _check_config(config):
    for name in ("patch_radius", "search_radius", "max_corners"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.patch_radius < 2:
        raise ValueError("Require a patch with measured two-dimensional context")
    if config.search_radius < math.ceil(config.geometry.max_displacement_px) + 2:
        raise ValueError("NCC search must exceed the static acceptance tolerance")
    if not 0 < config.corner_quality < 1 or not -1 < config.min_correlation <= 1:
        raise ValueError("Invalid corner quality or NCC correlation")
    for name in ("min_patch_std", "min_peak_gap", "max_reverse_error"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(config.peak_exclusion_radius, bool) or not isinstance(config.peak_exclusion_radius, int) or config.peak_exclusion_radius < 0:
        raise ValueError("Peak exclusion radius must be a nonnegative integer")


def _peak(response, config, integer_patch_correlation=None):
    """Require one distinct measured NCC peak, without a displacement filter."""
    if not np.isfinite(response).all():
        return None, {"reason": "nonfinite-ncc-response"}
    iy, ix = np.unravel_index(int(np.argmax(response)), response.shape)
    score = float(response[iy, ix])
    alternate = response.copy()
    r = config.peak_exclusion_radius
    alternate[max(0, iy-r):iy+r+1, max(0, ix-r):ix+r+1] = -np.inf
    finite = alternate[np.isfinite(alternate)]
    second = float(np.max(finite)) if len(finite) else -1.0
    gap = score - second
    stats = {"correlation": score, "secondCorrelation": second, "peakGap": gap}
    if integer_patch_correlation is not None:
        stats["integerPatchCorrelationFloat64"] = float(integer_patch_correlation)
    if score < config.min_correlation:
        return None, {**stats, "reason": "low-patch-correlation"}
    if gap < config.min_peak_gap:
        return None, {**stats, "reason": "ambiguous-repeated-patch"}
    # A maximum against the search boundary could be an unseen larger motion.
    if ix == 0 or iy == 0 or ix == response.shape[1]-1 or iy == response.shape[0]-1:
        return None, {**stats, "reason": "motion-reaches-search-boundary"}
    offsets = []
    for before, middle, after in ((response[iy, ix-1], response[iy, ix], response[iy, ix+1]),
                                   (response[iy-1, ix], response[iy, ix], response[iy+1, ix])):
        denominator = float(before - 2*middle + after)
        # OpenCV's float32 integral calculation can report slightly less than
        # one even for an exact linear-brightness integer correspondence. Test
        # that integer patch directly in float64, rather than widening a pixel
        # acceptance gate or fitting asymmetric neighbouring patch scores.
        perfect_integer = (integer_patch_correlation is not None
                           and integer_patch_correlation >= 1.0-1e-10)
        offset = (0.5 * float(before-after) / denominator
                  if config.subpixel and not perfect_integer and denominator < -1e-8 else 0.0)
        offsets.append(float(np.clip(offset, -0.5, 0.5)))
    stats["subpixelOffset"] = offsets
    return np.array([ix+offsets[0], iy+offsets[1]], np.float64), stats


def _match(gray_template, gray_search, center, static, config):
    r, s = config.patch_radius, config.search_radius
    center = np.asarray(center, np.float64)
    cx, cy = np.rint(center).astype(int)
    h, w = gray_search.shape
    padding = r+s+1
    if min(cx, cy, w-1-cx, h-1-cy) < padding:
        return None, {"reason": "incomplete-search-context"}
    # Both the interpolated template and every candidate target patch require
    # complete static support. A dark or absent detection is never sufficient.
    if not static[cy-padding:cy+padding+1, cx-padding:cx+padding+1].all():
        return None, {"reason": "nonstatic-search-context"}
    template = cv2.getRectSubPix(gray_template, (2*r+1, 2*r+1), tuple(map(float, center)))
    deviation = float(np.std(template))
    if deviation < config.min_patch_std:
        return None, {"reason": "insufficient-patch-texture", "patchStd": deviation}
    left, top = cx-s-r, cy-s-r
    search = gray_search[top:cy+s+r+1, left:cx+s+r+1]
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    peak_y, peak_x = np.unravel_index(int(np.argmax(response)), response.shape)
    actual_patch = search[peak_y:peak_y+2*r+1, peak_x:peak_x+2*r+1]
    left_values, right_values = template.astype(np.float64), actual_patch.astype(np.float64)
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    normalization = float(np.linalg.norm(left_values)*np.linalg.norm(right_values))
    exact_correlation = float(np.sum(left_values*right_values)/normalization) if normalization > 1e-12 else -1.0
    location, stats = _peak(response, config, exact_correlation)
    stats["patchStd"] = deviation
    if location is None:
        return None, stats
    return location + [left+r, top+r], stats


def unique_ncc_correspondences(first, second, static, config=NCCConfig()):
    """Audit all quality rejections and all symmetric unique correspondences."""
    _check_config(config)
    a, b, support = np.asarray(first), np.asarray(second), np.asarray(static)
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3 or a.shape != b.shape:
        raise ValueError("Require two uint8 RGB images in the same actual map pixel frame")
    if support.dtype != np.bool_ or support.shape != a.shape[:2]:
        raise ValueError("Require a boolean independently static support in the common frame")
    first_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    second_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    margin = config.patch_radius + config.search_radius + 1
    permitted = cv2.erode(support.astype(np.uint8), np.ones((2*margin+1, 2*margin+1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    corners = cv2.goodFeaturesToTrack(first_gray, config.max_corners, config.corner_quality,
                                     config.geometry.feature_spacing_px, mask=permitted*255, blockSize=3)
    candidates = [] if corners is None else [np.rint(p[0]).astype(np.float64) for p in corners]
    measured, rejected = [], []
    for origin in candidates:
        target, forward = _match(first_gray, second_gray, origin, support, config)
        record = {"first": origin.tolist(), "forward": forward}
        if target is None:
            rejected.append({**record, "reason": forward["reason"]})
            continue
        returned, reverse = _match(second_gray, first_gray, target, support, config)
        record.update(second=target.tolist(), reverse=reverse)
        if returned is None:
            rejected.append({**record, "reason": "reverse-"+reverse["reason"]})
            continue
        reverse_error = float(np.linalg.norm(returned-origin))
        record["reverseErrorPixels"] = reverse_error
        if reverse_error > config.max_reverse_error:
            rejected.append({**record, "reason": "asymmetric-patch-match"})
            continue
        measured.append(record)
    unique = []
    for record in sorted(measured, key=lambda r: (-min(r["forward"]["correlation"], r["reverse"]["correlation"]),
                                                 r["first"], r["second"])):
        x, y = np.asarray(record["first"]), np.asarray(record["second"])
        if any(np.linalg.norm(x-old[0]) < config.geometry.feature_spacing_px or
               np.linalg.norm(y-old[1]) < config.geometry.feature_spacing_px for old in unique):
            rejected.append({**record, "reason": "nonunique-spatial-corner"})
            continue
        unique.append((x, y, record))
    empty = np.empty((0, 2), np.float64)
    return (np.asarray([r[0] for r in unique]) if unique else empty,
            np.asarray([r[1] for r in unique]) if unique else empty,
            {"provider": "gftt-symmetric-unique-ncc", "candidateCorners": len(candidates),
             "qualityAcceptedBeforeSpatialUniqueness": len(measured), "unique": len(unique),
             "qualityAcceptedMeasurements": [r[2] for r in unique], "qualityRejections": rejected,
             "displacementUsedForMatchSelection": False, "searchRadiusPixels": config.search_radius})


def evidence_from_correspondences(a, b, shape, geometry=GeometryConfig()):
    """The unchanged geometric tolerances, applied to every measured holdout."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require finite matched Nx2 correspondence arrays")
    result = {"accepted": False, "reason": "insufficient-independent-matches",
              "unique": len(a), "fittedCorrectionApplied": False,
              "allQualityAcceptedHoldoutEvaluated": True}
    if len(a) < geometry.min_matches:
        return result
    h, w = shape
    if min(h, w) < 1 or (a < 0).any() or (a >= [w, h]).any() or (b < 0).any() or (b >= [w, h]).any():
        raise ValueError("Correspondences must remain in the actual common map frame")
    shifts = b-a
    cells = np.minimum((a/[w, h]*4).astype(int), 3)
    identifiers = cells[:, 1]*4+cells[:, 0]
    train = (cells[:, 0]+cells[:, 1]) % 2 == 0
    heldout = ~train
    result.update(trainingPoints=int(train.sum()), heldoutPoints=int(heldout.sum()))
    if train.sum() < 4 or heldout.sum() < 4:
        result["reason"] = "insufficient-independent-holdout"
        return result
    training_shifts = shifts[train]
    counts = [int((np.linalg.norm(training_shifts-value, axis=1) <= geometry.consensus_radius_px).sum())
              for value in training_shifts]
    initial = training_shifts[int(np.argmax(counts))]
    fitting = np.linalg.norm(training_shifts-initial, axis=1) <= geometry.consensus_radius_px
    translation = np.median(training_shifts[fitting], axis=0)
    inliers = np.linalg.norm(shifts-translation, axis=1) <= geometry.consensus_radius_px
    occupied = sorted(set(identifiers[inliers].tolist()))
    span = np.ptp(a[inliers], axis=0)/[w, h]
    # No consensus/outlier filter is applied to the independent measurements.
    residual = np.linalg.norm(shifts[heldout]-translation, axis=1)
    displacement = np.linalg.norm(shifts[heldout], axis=1)
    p90 = float(np.percentile(displacement, 90))
    holdout_p90 = float(np.percentile(residual, 90))
    result.update(inliers=int(inliers.sum()), consensusFraction=float(inliers.mean()),
                  occupiedCells=occupied, spanFractions=span.tolist(), medianVectorPixels=translation.tolist(),
                  p90ActualPixels=p90, holdoutResidualP90Pixels=holdout_p90,
                  correspondences=[{"first": x.tolist(), "second": y.tolist(), "delta": d.tolist(),
                                    "inlier": bool(ok), "heldout": bool(test)}
                                   for x, y, d, ok, test in zip(a, b, shifts, inliers, heldout, strict=True)])
    reason = "verified"
    if inliers.sum() < geometry.min_matches or inliers.mean() < geometry.min_consensus_fraction:
        reason = "insufficient-consensus"
    elif len(occupied) < geometry.min_cells or max(span) < 0.35 or min(span) < 0.08:
        reason = "spatially-concentrated-features"
    elif holdout_p90 > geometry.consensus_radius_px:
        reason = "independent-holdout-failed"
    elif p90 > geometry.max_displacement_px:
        reason = "actual-static-displacement"
    result.update(accepted=reason == "verified", reason=reason)
    return result


def gftt_ncc_static_evidence(first, second, static, config=NCCConfig()):
    a, b, audit = unique_ncc_correspondences(first, second, static, config)
    proof = evidence_from_correspondences(a, b, np.asarray(static).shape, config.geometry)
    return {**audit, **proof, "providerConfig": asdict(config),
            "pixelFrame": "actual unresized input orthophoto grid", "noWarp": True,
            "noGeneratedBackground": True, "measurementSelectionUsesAcceptanceTolerance": False}


def compare_static_providers(first, second, static, config=NCCConfig()):
    """Keep both providers visible; this report makes no automatic gate choice."""
    sift = independent_static_evidence(first, second, static, config.geometry)
    ncc = gftt_ncc_static_evidence(first, second, static, config)
    return {"schemaVersion": "ogs2-corridor-static-provider-comparison/v1", "providers": {"sift": sift, "gfttNcc": ncc},
            "automaticProviderPromotion": False, "fittedCorrectionApplied": False,
            "acceptanceDisagreement": bool(sift["accepted"] != ncc["accepted"]),
            "reasons": {"sift": sift["reason"], "gfttNcc": ncc["reason"]}}
