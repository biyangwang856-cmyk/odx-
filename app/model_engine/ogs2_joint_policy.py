"""CPU policy draft for whole-time vehicle support and static-only seam collars.

This module does not infer identities, shadows, calibration, or visibility.
Its caller must provide those observations in one calibrated map pixel frame.
It prevents cheap empty images and unrecorded foreground/background blending.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SparseRegion:
    origin: tuple[int, int]
    mask: np.ndarray

    def __post_init__(self):
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("Require a nonempty two-dimensional support mask")
        if len(self.origin) != 2 or any(int(v) != v for v in self.origin):
            raise ValueError("Support origins must be integer map pixels")
        object.__setattr__(self, "origin", tuple(int(v) for v in self.origin))
        object.__setattr__(self, "mask", mask.copy())

    @property
    def bounds(self):
        x, y = self.origin
        return x, y, x + self.mask.shape[1], y + self.mask.shape[0]


@dataclass(frozen=True)
class PatchPlan:
    identity: str
    support: SparseRegion
    required_tracks: frozenset[str]
    complete_tracks_by_source: Mapping[str, frozenset[str]]
    clock: str
    first_time: float
    last_time: float


@dataclass(frozen=True)
class JointPlan:
    members: tuple[str, ...]
    support: SparseRegion | None
    required_tracks: frozenset[str]
    source_candidates: tuple[str, ...]
    rejection_reason: str | None = None


@dataclass(frozen=True)
class CandidateEvidence:
    source_image: str
    valid: np.ndarray
    observed_track_masks: Mapping[str, np.ndarray]
    complete_tracks: frozenset[str]
    static_verified: bool
    boundary_clear: bool
    cost: float
    observation_ids_by_track: Mapping[str, tuple[str, ...]] | None = None


def regions_overlap(first: SparseRegion, second: SparseRegion) -> bool:
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(
        np.any(
            first.mask[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
            & second.mask[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
        )
    )


def merge_conflicts(
    plans: list[PatchPlan], max_pixels: int = 1_000_000, max_span_seconds: float = 3.0
) -> list[JointPlan]:
    """Union actual intersecting support masks before any raster write.

    A support should include the reserved seam collar so separate writers cannot
    modify one another's collar. A vehicle identity remains separate from the
    resulting geometric patch component. Cross-clock or long-time components
    are explicitly withheld rather than silently selecting an arbitrary time.
    """
    if max_pixels <= 0 or max_span_seconds <= 0:
        raise ValueError("Require positive support budget and time span")
    if len({plan.identity for plan in plans}) != len(plans):
        raise ValueError("Patch identities must be unique")
    parents = list(range(len(plans)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first in range(len(plans)):
        for second in range(first + 1, len(plans)):
            if regions_overlap(plans[first].support, plans[second].support):
                parents[find(second)] = find(first)
    components = {}
    for index in range(len(plans)):
        components.setdefault(find(index), []).append(plans[index])
    result = []
    for component in components.values():
        component.sort(key=lambda plan: plan.identity)
        identities = tuple(plan.identity for plan in component)
        required = frozenset(track for plan in component for track in plan.required_tracks)
        reason = None
        clocks = {plan.clock for plan in component}
        times = [value for plan in component for value in (plan.first_time, plan.last_time)]
        if not required:
            reason = "no-required-vehicle"
        elif len(clocks) != 1 or not next(iter(clocks)) or "unknown" in clocks:
            reason = "incompatible-clocks"
        elif not all(math.isfinite(value) for value in times):
            reason = "unknown-capture-time"
        elif any(plan.last_time < plan.first_time for plan in component):
            reason = "invalid-time-interval"
        elif max(times) - min(times) > max_span_seconds:
            reason = "incompatible-temporal-cohorts"
        x0 = min(plan.support.bounds[0] for plan in component)
        y0 = min(plan.support.bounds[1] for plan in component)
        x1 = max(plan.support.bounds[2] for plan in component)
        y1 = max(plan.support.bounds[3] for plan in component)
        if (x1 - x0) * (y1 - y0) > max_pixels:
            reason = reason or "joint-support-exceeds-budget"
        coverage = {}
        for plan in component:
            for source, tracks in plan.complete_tracks_by_source.items():
                coverage.setdefault(source, set()).update(tracks)
        sources = tuple(sorted(source for source, tracks in coverage.items() if required <= tracks))
        if not sources:
            reason = reason or "no-common-whole-track-source"
        if reason:
            result.append(JointPlan(identities, None, required, (), reason))
            continue
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        for plan in component:
            x, y = plan.support.origin
            h, w = plan.support.mask.shape
            mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] |= plan.support.mask
        result.append(JointPlan(identities, SparseRegion((x0, y0), mask), required, sources))
    return sorted(result, key=lambda plan: plan.members)


def choose_whole_source(
    hard_support: np.ndarray,
    required_tracks: frozenset[str],
    candidates: list[CandidateEvidence],
    min_vehicle_pixels: int = 16,
) -> tuple[str | None, list[dict]]:
    """Require every protected identity in the same real, fully valid photo.

    The hard support is the union of old/new vehicle, shadow, blur and residual
    support, not merely the selected vehicle mask. It remains single-source.
    """
    hard = np.asarray(hard_support, bool)
    if hard.ndim != 2 or not hard.any() or not required_tracks:
        raise ValueError("A vehicle-bearing nonempty hard support is required")
    if min_vehicle_pixels <= 0:
        raise ValueError("Require a positive minimum observed vehicle footprint")
    eligible, decisions = [], []
    for candidate in candidates:
        reason = None
        valid = np.asarray(candidate.valid, bool)
        if valid.shape != hard.shape:
            raise ValueError("Candidate validity uses a different pixel frame")
        if not math.isfinite(candidate.cost):
            reason = "invalid-source-cost"
        elif not valid[hard].all():
            reason = "incomplete-dynamic-support-coverage"
        elif not required_tracks <= candidate.complete_tracks:
            reason = "missing-complete-protected-vehicle"
        elif candidate.observation_ids_by_track is None:
            reason = "unverified-vehicle-instance-count"
        elif any(
            len(candidate.observation_ids_by_track.get(identity, ())) != 1
            for identity in required_tracks
        ):
            reason = "protected-vehicle-not-observed-exactly-once"
        elif len(
            {
                candidate.observation_ids_by_track[identity][0]
                for identity in required_tracks
            }
        ) != len(required_tracks):
            reason = "different-identities-share-one-observation"
        elif not candidate.static_verified:
            reason = "unverified-static-geometry"
        elif not candidate.boundary_clear:
            reason = "unsafe-foreground-boundary"
        else:
            for identity in sorted(required_tracks):
                observed = candidate.observed_track_masks.get(identity)
                if observed is None:
                    reason = "missing-observed-vehicle-core"
                    break
                observed = np.asarray(observed, bool)
                if observed.shape != hard.shape:
                    raise ValueError("Vehicle evidence uses a different pixel frame")
                if observed.sum() < min_vehicle_pixels:
                    reason = "empty-or-insufficient-observed-vehicle"
                    break
                if (observed & ~hard).any():
                    reason = "vehicle-core-outside-hard-support"
                    break
        if reason is None:
            eligible.append((candidate.cost, candidate.source_image))
        decisions.append({"sourceImage": candidate.source_image, "reason": reason or "eligible"})
    return (min(eligible)[1] if eligible else None), decisions


def compose_static_collar(
    baseline_rgba: np.ndarray,
    source_rgb: np.ndarray,
    hard_support: np.ndarray,
    source_weight: np.ndarray,
    verified_static: np.ndarray,
    valid_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix only observed, geometrically verified static background pixels.

    The caller provides a smooth collar weight; no object, shadow or unknown
    dynamic pixel may receive a fractional weight. Mode 0 is baseline, 1 is
    unblended source and 2 is static mixing. Save the returned float weight so a
    mixed background pixel never masquerades as a pure source-index pixel.
    """
    baseline = np.asarray(baseline_rgba)
    source = np.asarray(source_rgb)
    if baseline.dtype != np.uint8 or source.dtype != np.uint8:
        raise ValueError("Require explicit uint8 RGB and baseline alpha")
    if baseline.ndim != 3 or baseline.shape[2] != 4 or source.shape != (*baseline.shape[:2], 3):
        raise ValueError("Require HxWx4 baseline and HxWx3 source in the same grid")
    hard = np.asarray(hard_support, bool)
    static = np.asarray(verified_static, bool)
    valid = np.asarray(valid_source, bool)
    weight = np.asarray(source_weight, np.float32)
    if any(value.shape != baseline.shape[:2] for value in (hard, static, valid, weight)):
        raise ValueError("Masks and source weight use a different pixel frame")
    if not hard.any() or not np.isfinite(weight).all() or ((weight < 0) | (weight > 1)).any():
        raise ValueError("Require nonempty hard support and finite weights in [0,1]")
    if not (weight[hard] == 1).all():
        raise ValueError("Vehicle/shadow hard support cannot be blended or erased")
    written, mixed = weight > 0, (weight > 0) & (weight < 1)
    if (written & (~valid | (baseline[..., 3] == 0))).any():
        raise ValueError("No partial coverage or expansion beyond baseline is allowed")
    if (mixed & (~static | hard)).any():
        raise ValueError("Only verified static background may be fractionally blended")
    if (written & ~hard & ~static).any():
        raise ValueError("Unknown foreground outside hard support must be expanded or rejected")
    result = baseline.copy()
    colors = weight[..., None] * source.astype(np.float32)
    colors += (1 - weight[..., None]) * baseline[..., :3].astype(np.float32)
    result[written, :3] = np.rint(colors[written]).clip(0, 255).astype(np.uint8)
    mode = np.zeros(hard.shape, np.uint8)
    mode[written] = 1
    mode[mixed] = 2
    return result, mode, weight.copy()
