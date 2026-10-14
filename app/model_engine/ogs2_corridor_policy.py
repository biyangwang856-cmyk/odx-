"""A bounded road region takes one observed exposure, including real empty road.

This contract does not establish vehicle identity or a global capture instant.
It permits replacing historical vehicle pixels only where the selected real
exposure has independently verified visible road. Complete cars and their
observed shadows are indivisible foregrounds. No inferred RGB is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from .ogs2_joint_closure import close_whole_foregrounds


def array_sha256(value: np.ndarray) -> str:
    """Bind an evidence record to dtype, shape and exact common-frame pixels."""
    value = np.asarray(value)
    digest = sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CorridorConfig:
    max_context_pixels: int = 4_000_000
    max_support_pixels: int = 2_000_000
    protection_steps: int = 1
    static_collar_steps: int = 2
    min_static_pixels: int = 48
    max_static_displacement_px: float = 2.0
    max_holdout_residual_px: float = 1.0
    min_static_matches: int = 12
    min_static_cells: int = 4
    min_consensus_fraction: float = 0.65


@dataclass(frozen=True)
class CorridorCandidate:
    source_id: str
    rgb: np.ndarray
    valid: np.ndarray
    observed_road: np.ndarray
    verified_empty_road: np.ndarray
    static_support: np.ndarray
    foregrounds: Mapping[str, np.ndarray]
    complete_foregrounds: frozenset[str]
    cost: float
    capture_time: float
    static_verification: Mapping[str, object]
    unknown_foreground: np.ndarray
    foreground_catalog_complete: bool = False


@dataclass(frozen=True)
class CorridorPlan:
    source: CorridorCandidate
    hard_support: np.ndarray
    static_collar: np.ndarray
    included_foregrounds: tuple[str, ...]
    report: Mapping[str, object] = field(default_factory=dict)
    validation_sha256: str = ""


def _mask(value, shape, name):
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask in the common pixel frame")
    return mask


def _dilate(mask, steps):
    result = mask.copy()
    for _ in range(steps):
        old = result.copy()
        result[1:] |= old[:-1]
        result[:-1] |= old[1:]
        result[:, 1:] |= old[:, :-1]
        result[:, :-1] |= old[:, 1:]
    return result


def _boundary(shape):
    result = np.zeros(shape, bool)
    result[[0, -1], :] = True
    result[:, [0, -1]] = True
    return result


def _validate_config(config):
    positive = ("max_context_pixels", "max_support_pixels", "min_static_pixels", "min_static_matches", "min_static_cells")
    for name in positive:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("protection_steps", "static_collar_steps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config.static_collar_steps < 1:
        raise ValueError("Require at least one pixel of independent static context")
    for name in ("max_static_displacement_px", "max_holdout_residual_px"):
        value = getattr(config, name)
        if not _finite_scalar(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not _finite_scalar(config.min_consensus_fraction) or not 0 < config.min_consensus_fraction <= 1:
        raise ValueError("min_consensus_fraction must be finite in (0, 1]")


def _finite_scalar(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _plan_sha256(candidate, hard, collar):
    """Detect accidental mutation of the accepted pixel/mask/evidence snapshot."""
    digest = sha256()
    for value in (candidate.rgb, candidate.valid, candidate.observed_road,
                  candidate.verified_empty_road, candidate.static_support,
                  candidate.unknown_foreground, hard, collar):
        digest.update(array_sha256(value).encode("ascii"))
    for name, mask in sorted(candidate.foregrounds.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(mask).encode("ascii"))
    digest.update(repr((candidate.source_id, candidate.cost, candidate.capture_time,
                        candidate.foreground_catalog_complete,
                        sorted(candidate.complete_foregrounds))).encode("utf-8"))
    # Only the proof fields evaluated by this contract are included. Its optional
    # correspondences can be huge; its image/static hashes already bind input.
    keys = ("accepted", "sourceId", "sourceImageSha256", "baselineImageSha256",
            "baselineRgbaSha256", "staticSupportSha256", "referenceKind", "trainingPoints", "heldoutPoints",
            "inliers", "p90ActualPixels", "holdoutResidualP90Pixels", "occupiedCells", "spanFractions",
            "consensusFraction", "fittedCorrectionApplied")
    digest.update(json.dumps({k: candidate.static_verification.get(k) for k in keys},
                             sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)).encode("utf-8"))
    return digest.hexdigest()


def _geometry_reason(candidate, baseline, config):
    proof = candidate.static_verification
    if not isinstance(proof, Mapping) or proof.get("accepted") is not True:
        return "baseline-static-proof-not-accepted"
    bindings = {
        "sourceId": candidate.source_id,
        "sourceImageSha256": array_sha256(candidate.rgb),
        "baselineImageSha256": array_sha256(baseline[..., :3]),
        "baselineRgbaSha256": array_sha256(baseline),
        "staticSupportSha256": array_sha256(candidate.static_support),
    }
    if any(proof.get(key) != expected for key, expected in bindings.items()):
        return "baseline-static-proof-binding-mismatch"
    # Source-to-source proof alone cannot authorize the actual baseline seam.
    if proof.get("referenceKind") != "orthophoto-baseline":
        return "baseline-static-reference-required"
    fraction = proof.get("consensusFraction")
    if not _finite_scalar(fraction) or not config.min_consensus_fraction <= fraction <= 1:
        return "insufficient-static-consensus"
    if proof.get("fittedCorrectionApplied") is not False:
        return "warped-static-proof-is-not-permitted"
    for name, minimum in (("trainingPoints", 4), ("heldoutPoints", 4), ("inliers", config.min_static_matches)):
        value = proof.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            return "insufficient-independent-static-matches"
    for name, limit in (("p90ActualPixels", config.max_static_displacement_px),
                        ("holdoutResidualP90Pixels", config.max_holdout_residual_px)):
        value = proof.get(name)
        if not _finite_scalar(value) or not 0 <= value <= limit:
            return "actual-baseline-static-displacement"
    cells = proof.get("occupiedCells")
    if not isinstance(cells, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not 0 <= x < 16 for x in cells):
        return "invalid-static-cell-support"
    if len(set(cells)) < config.min_static_cells:
        return "spatially-concentrated-static-evidence"
    span = np.asarray(proof.get("spanFractions", []))
    if span.shape != (2,) or not np.issubdtype(span.dtype, np.number) or not np.isfinite(span).all():
        return "invalid-static-span"
    if np.min(span) < 0.08 or np.max(span) < 0.35 or np.max(span) > 1:
        return "spatially-concentrated-static-evidence"
    return None


def select_corridor_source(
    baseline_rgba: np.ndarray,
    conflict_union: np.ndarray,
    road_domain: np.ndarray,
    candidates: Sequence[CorridorCandidate],
    *,
    baseline_foregrounds: Mapping[str, np.ndarray] | None = None,
    protected_unknown: np.ndarray,
    config: CorridorConfig = CorridorConfig(),
) -> tuple[CorridorPlan | None, list[dict]]:
    """Select by real source quality, never by a reduced vehicle count.

    `conflict_union` is a whole road conflict region, not a requested vehicle
    identity. It must contain all historical appearances slated for replacement.
    The orchestrator supplies source-specific observed empty-road support and
    a hash-bound held-out baseline static proof. No detection is not that proof.
    Every unknown/parked foreground supplied by the caller remains protected.

    Bad input raises; a physically unsupported candidate is rejected atomically
    and recorded. No partial coverage, partial object or relaxed guard is used.
    """
    _validate_config(config)
    baseline = np.asarray(baseline_rgba)
    if baseline.dtype != np.uint8 or baseline.ndim != 3 or baseline.shape[2] != 4:
        raise ValueError("Require a uint8 HxWx4 baseline")
    shape = baseline.shape[:2]
    if min(shape) < 3 or np.prod(shape) > config.max_context_pixels:
        raise ValueError("Require bounded context with an interior")
    conflict = _mask(conflict_union, shape, "conflict_union")
    road = _mask(road_domain, shape, "road_domain")
    unknown = _mask(protected_unknown, shape, "protected_unknown")
    if not conflict.any():
        raise ValueError("Require a nonempty road conflict region")
    original_foregrounds = {
        name: _mask(mask, shape, f"baseline foreground {name}")
        for name, mask in (baseline_foregrounds or {}).items()
    }
    if any(not isinstance(name, str) or not name for name in original_foregrounds):
        raise ValueError("Foreground observation IDs must be nonempty strings")
    if any(not isinstance(c, CorridorCandidate) for c in candidates):
        raise ValueError("Require CorridorCandidate observations")
    if len({c.source_id for c in candidates}) != len(candidates):
        raise ValueError("Source IDs must be unique")
    covered = baseline[..., 3] > 0
    edge = _boundary(shape)
    decisions, eligible = [], []
    for candidate in candidates:
        if not isinstance(candidate.source_id, str) or not candidate.source_id:
            raise ValueError("Require a nonempty source ID")
        rgb = np.asarray(candidate.rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
            raise ValueError("Candidate RGB must be uint8 in the baseline pixel frame")
        fields = {name: _mask(getattr(candidate, name), shape, name)
                  for name in ("valid", "observed_road", "verified_empty_road", "static_support", "unknown_foreground")}
        if not _finite_scalar(candidate.cost) or not _finite_scalar(candidate.capture_time):
            raise ValueError("Source cost and capture time must be finite")
        foregrounds = {name: _mask(mask, shape, f"source foreground {name}")
                       for name, mask in candidate.foregrounds.items()}
        if any(not isinstance(name, str) or not name or not mask.any() for name, mask in foregrounds.items()):
            raise ValueError("Every foreground requires a nonempty observation ID and mask")
        if not isinstance(candidate.complete_foregrounds, (set, frozenset)) or not candidate.complete_foregrounds.issubset(foregrounds):
            raise ValueError("Complete foreground claim has no actual source mask")
        catalog = {f"baseline:{name}": mask for name, mask in original_foregrounds.items()}
        catalog.update({f"source:{name}": mask for name, mask in foregrounds.items()})
        decision = {"sourceId": candidate.source_id, "cost": float(candidate.cost),
                    "captureTime": float(candidate.capture_time), "accepted": False}
        reason = None
        try:
            closure = close_whole_foregrounds(conflict, catalog, config.protection_steps, config.max_support_pixels)
            hard = closure.support
        except ValueError:
            reason = "whole-foreground-closure-exceeds-budget"
            hard = conflict
        touching = {name: mask for name, mask in foregrounds.items() if (mask & hard).any()}
        source_union = np.logical_or.reduce(list(touching.values())) if touching else np.zeros(shape, bool)
        collar = _dilate(hard, config.static_collar_steps) & ~hard
        if reason is None:
            if candidate.foreground_catalog_complete is not True:
                reason = "source-foreground-catalog-is-incomplete"
            elif (hard & edge).any() or (collar & edge).any():
                reason = "expand-context-boundary-is-unknown"
            elif (hard & ~road).any():
                reason = "whole-support-outside-approved-road"
            elif (hard & (unknown | fields["unknown_foreground"])).any():
                reason = "protected-unknown-or-parked-foreground"
            elif (hard & (~covered | ~fields["valid"])).any():
                reason = "incomplete-real-source-or-baseline-coverage"
            elif any(name not in candidate.complete_foregrounds or (mask & edge).any() for name, mask in touching.items()):
                reason = "incomplete-source-foreground"
            elif (hard & ~source_union & ~(fields["observed_road"] & fields["verified_empty_road"])).any():
                reason = "old-position-has-no-verified-observed-empty-road"
            elif (collar & (~fields["static_support"] | ~fields["valid"] | ~covered | unknown | fields["unknown_foreground"])).any():
                reason = "unverified-static-collar"
            elif any((mask & collar).any() and not (mask & hard).any() for mask in catalog.values()):
                reason = "static-collar-touches-unselected-foreground"
            elif int(collar.sum()) < config.min_static_pixels:
                reason = "insufficient-static-collar-support"
            else:
                reason = _geometry_reason(candidate, baseline, config)
        decision.update(accepted=reason is None, reason=reason or "verified-one-exposure-road-region",
                        supportPixels=int(hard.sum()), selectedForegroundObservations=len(touching))
        decisions.append(decision)
        if reason is None:
            report = {**decision, "staticCollarPixels": int(collar.sum()),
                      "emptyRoadPixels": int((hard & ~source_union).sum()),
                      "vehicleCountUsedForRanking": False, "identityAssociationAssumed": False,
                      "globalInstantClaimed": False, "rgbGeneration": False, "hardRegionFeathering": False}
            eligible.append(CorridorPlan(candidate, hard.copy(), collar.copy(), closure.included_foregrounds,
                                         report, _plan_sha256(candidate, hard, collar)))
    if not eligible:
        return None, decisions
    selected = min(eligible, key=lambda p: (p.source.cost, p.source.capture_time, p.source.source_id))
    return selected, decisions


def compose_corridor(baseline_rgba: np.ndarray, plan: CorridorPlan) -> tuple[np.ndarray, np.ndarray, dict]:
    """Copy actual selected-source RGB exactly; preserve alpha and other pixels."""
    baseline = np.asarray(baseline_rgba)
    if baseline.dtype != np.uint8 or baseline.ndim != 3 or baseline.shape[2] != 4:
        raise ValueError("Require uint8 HxWx4 baseline")
    shape = baseline.shape[:2]
    hard = _mask(plan.hard_support, shape, "plan hard support")
    rgb = np.asarray(plan.source.rgb)
    valid = _mask(plan.source.valid, shape, "source valid")
    proof = plan.source.static_verification
    if not plan.validation_sha256 or plan.validation_sha256 != _plan_sha256(plan.source, hard, plan.static_collar):
        raise ValueError("Accepted support/source/evidence changed after planning")
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError("Source RGB uses another frame")
    if not hard.any() or (hard & (~valid | (baseline[..., 3] == 0))).any():
        raise ValueError("Cannot compose unknown/empty support")
    if proof.get("sourceImageSha256") != array_sha256(rgb) or proof.get("baselineImageSha256") != array_sha256(baseline[..., :3]) or proof.get("baselineRgbaSha256") != array_sha256(baseline):
        raise ValueError("Source/baseline pixels changed after planning")
    output = baseline.copy()
    output[..., :3][hard] = rgb[hard]
    mode = hard.astype(np.uint8)
    if not np.array_equal(output[..., 3], baseline[..., 3]) or not np.array_equal(output[~hard], baseline[~hard]):
        raise RuntimeError("Corridor composition changed alpha or unsupported pixels")
    return output, mode, {**plan.report, "writtenPixels": int(hard.sum()), "alphaChangedPixels": 0,
                          "outsideSupportChangedPixels": 0, "sourceId": plan.source.source_id}
