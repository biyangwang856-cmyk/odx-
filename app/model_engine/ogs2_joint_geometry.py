"""Independent static correspondences and observed dynamic support.

No camera fitting, image warp, learned background, or geometry modification.
All inputs are already in one calibrated orthophoto pixel frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GeometryConfig:
    ratio: float = 0.7
    feature_spacing_px: float = 3.0
    min_matches: int = 12
    min_cells: int = 4
    max_displacement_px: float = 2.0
    consensus_radius_px: float = 1.0
    min_consensus_fraction: float = 0.65
    max_dynamic_distance_m: float = 7.0
    dynamic_difference: float = 25.0
    shadow_difference: float = 15.0
    noise_component_pixels: int = 8


def unique_correspondences(first, second, static, config=GeometryConfig()):
    """Mutual ratio matches, unique indices and spatially separate features."""
    static = np.asarray(static, bool)
    if first.shape != second.shape or first.shape[:2] != static.shape:
        raise ValueError("Static comparison must use a common pixel frame")
    extractor = cv2.SIFT_create(nfeatures=6000)
    mask = static.astype(np.uint8) * 255
    a, da = extractor.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), mask)
    b, db = extractor.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), mask)
    empty = np.empty((0, 2), np.float64)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return empty, empty, {"rawForward": 0, "rawMutual": 0, "unique": 0}
    matcher = cv2.BFMatcher()

    def ratio_pairs(left, right):
        return {
            m.queryIdx: m
            for pair in matcher.knnMatch(left, right, k=2)
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < config.ratio * n.distance
        }

    forward, backward = ratio_pairs(da, db), ratio_pairs(db, da)
    mutual = [
        m for m in forward.values() if m.trainIdx in backward and backward[m.trainIdx].trainIdx == m.queryIdx
    ]
    accepted = []
    for match in sorted(mutual, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        x = np.asarray(a[match.queryIdx].pt)
        y = np.asarray(b[match.trainIdx].pt)
        # SIFT may emit multiple orientations at the same physical corner.
        if any(
            np.linalg.norm(x - old_x) < config.feature_spacing_px
            or np.linalg.norm(y - old_y) < config.feature_spacing_px
            for old_x, old_y in accepted
        ):
            continue
        accepted.append((x, y))
    points = np.asarray(accepted, np.float64).reshape(-1, 2, 2)
    return (
        points[:, 0],
        points[:, 1],
        {"rawForward": len(forward), "rawMutual": len(mutual), "unique": len(accepted)},
    )


def independent_static_evidence(first, second, static, config=GeometryConfig()):
    """Reject isolated feature clusters and report all rejected correspondences.

    Translation consensus is diagnostic only. Acceptance requires actual
    uncorrected displacement on spatially distributed held-out observations.
    """
    a, b, counts = unique_correspondences(first, second, static, config)
    result = {**counts, "accepted": False, "reason": "insufficient-independent-matches"}
    if len(a) < config.min_matches:
        return result
    shifts = b - a
    h, w = static.shape
    cells = np.minimum((a / [w, h] * 4).astype(int), 3)
    identifiers = cells[:, 1] * 4 + cells[:, 0]
    # Fixed checkerboard split precedes all model/consensus selection.
    train = (cells[:, 0] + cells[:, 1]) % 2 == 0
    heldout = ~train
    if train.sum() < 4 or heldout.sum() < 4:
        result.update(
            reason="insufficient-independent-holdout",
            trainingPoints=int(train.sum()),
            heldoutPoints=int(heldout.sum()),
        )
        return result
    training_shifts = shifts[train]
    centers = np.asarray(
        [
            int(np.sum(np.linalg.norm(training_shifts - value, axis=1) <= config.consensus_radius_px))
            for value in training_shifts
        ]
    )
    initial = training_shifts[int(np.argmax(centers))]
    fitting = np.linalg.norm(training_shifts - initial, axis=1) <= config.consensus_radius_px
    translation = np.median(training_shifts[fitting], axis=0)
    inliers = np.linalg.norm(shifts - translation, axis=1) <= config.consensus_radius_px
    occupied = sorted(set(identifiers[inliers].tolist()))
    span = np.ptp(a[inliers], axis=0) / [w, h]
    # Every original held-out match is evaluated, including its outliers.
    residual = np.linalg.norm(shifts[heldout] - translation, axis=1)
    displacement = np.linalg.norm(shifts[heldout], axis=1)
    p90 = float(np.percentile(displacement, 90))
    holdout_p90 = float(np.percentile(residual, 90)) if len(residual) else None
    result.update(
        {
            "inliers": int(inliers.sum()),
            "consensusFraction": float(inliers.mean()),
            "occupiedCells": occupied,
            "spanFractions": span.tolist(),
            "medianVectorPixels": translation.tolist(),
            "p90ActualPixels": p90,
            "trainingPoints": int(train.sum()),
            "heldoutPoints": int(heldout.sum()),
            "holdoutResidualP90Pixels": holdout_p90,
            "correspondences": [
                {"first": x.tolist(), "second": y.tolist(), "delta": d.tolist(), "inlier": bool(ok)}
                for x, y, d, ok in zip(a, b, shifts, inliers, strict=True)
            ],
            "fittedCorrectionApplied": False,
        }
    )
    reason = "verified"
    if inliers.sum() < config.min_matches or inliers.mean() < config.min_consensus_fraction:
        reason = "insufficient-consensus"
    elif len(occupied) < config.min_cells or max(span) < 0.35 or min(span) < 0.08:
        reason = "spatially-concentrated-features"
    elif train.sum() < 4 or heldout.sum() < 4 or holdout_p90 > config.consensus_radius_px:
        reason = "independent-holdout-failed"
    elif p90 > config.max_displacement_px:
        reason = "actual-static-displacement"
    result.update(accepted=reason == "verified", reason=reason)
    return result


def observed_dynamic_support(images, valids, body_union, pixel_size_m, config=GeometryConfig()):
    """Expand body seeds only through observed inter-frame appearance changes.

    Shadows are included as observed changes near the body, not synthesized
    semantic labels. Closing connects candidate components but its added pixels
    are never reported as measured change. A small protection dilation is
    explicitly returned separately from measured changes.
    """
    body = np.asarray(body_union, bool)
    if len(images) < 2 or len(images) != len(valids) or not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("Require at least two real frames and metric pixel scale")
    if any(image.shape != (*body.shape, 3) for image in images):
        raise ValueError("Images and masks must use a common pixel frame")
    if any(valid.shape != body.shape for valid in valids):
        raise ValueError("Validity and body masks use different grids")
    radius = config.max_dynamic_distance_m / pixel_size_m
    domain = ndimage.distance_transform_edt(~body) <= radius
    candidate = np.zeros(body.shape, bool)
    for index in range(len(images)):
        for other in range(index + 1, len(images)):
            delta = np.max(
                np.abs(images[index].astype(np.float32) - images[other].astype(np.float32)), axis=2
            )
            first_gray=cv2.cvtColor(images[index],cv2.COLOR_RGB2GRAY).astype(np.float32)
            other_gray=cv2.cvtColor(images[other],cv2.COLOR_RGB2GRAY).astype(np.float32)
            illumination=np.abs(first_gray-other_gray)
            candidate |= ((delta >= config.dynamic_difference) | (illumination>=config.shadow_difference)) & (
                valids[index] & valids[other] & domain
            )
    # Use connectivity solely to discover a support; do not claim synthetic
    # morphological pixels are measured shadows.
    joined = ndimage.binary_closing(candidate | body, iterations=2) & domain
    labels, count = ndimage.label(joined)
    touching = np.unique(labels[ndimage.binary_dilation(body, iterations=2)])
    touching = touching[touching != 0]
    measured = candidate & np.isin(labels, touching)
    parts, part_count = ndimage.label(measured)
    sizes = np.bincount(parts.ravel(), minlength=part_count + 1)
    measured &= sizes[parts] >= config.noise_component_pixels
    hard = ndimage.binary_dilation(body | measured, iterations=max(2, int(np.ceil(0.25 / pixel_size_m))))
    return (
        hard,
        measured,
        {
            "bodyPixels": int(body.sum()),
            "measuredChangePixels": int(measured.sum()),
            "outsideBodyMeasuredPixels": int((measured & ~body).sum()),
            "hardSupportPixels": int(hard.sum()),
            "maxDistanceMeters": config.max_dynamic_distance_m,
            "differenceThreshold": config.dynamic_difference,
            "interpretation": "Observed change support near body; not a perfect semantic shadow mask",
        },
    )


def close_unknown_boundary(hard, verified_static, valid, covered, max_pixels=1_000_000, margin=2):
    """Keep a whole uncertain transition on the source side of a static seam.

    Unknown components touching the source margin are indivisible. A component
    reaching unseen pixels is rejected rather than clipping an object/shadow.
    """
    hard = np.asarray(hard, bool).copy()
    static = np.asarray(verified_static, bool)
    valid = np.asarray(valid, bool)
    covered = np.asarray(covered, bool)
    if any(mask.shape != hard.shape for mask in (static, valid, covered)) or not hard.any():
        raise ValueError("Unknown closure requires matched, nonempty map masks")
    labels, count = ndimage.label(~static & ~hard, np.ones((3, 3), bool))
    edge = ndimage.binary_dilation(hard, iterations=margin) & ~hard
    touched = np.unique(labels[edge & ~static])
    touched = touched[touched != 0]
    added = np.isin(labels, touched)
    if (added & (~valid | ~covered)).any():
        raise ValueError("Unknown transition reaches unseen pixels; expand or reject")
    hard |= added
    hard = ndimage.binary_fill_holes(hard)
    if hard.sum() > max_pixels:
        raise ValueError("Whole unknown transition exceeds source budget")
    ring = ndimage.binary_dilation(hard, iterations=margin) & ~hard
    if (ring & ~static).any():
        raise ValueError("Uncertainty closure did not produce a verified static margin")
    return hard, {
        "unknownComponentsIncluded": int(len(touched)),
        "addedUncertainPixels": int(added.sum()),
        "hardPixelsAfterClosure": int(hard.sum()),
    }
