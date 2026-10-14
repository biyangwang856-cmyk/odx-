"""Explicit exposure choices for spatially connected, different-time patches.

This optional helper leaves ``merge_conflicts``' conservative three-second
contract unchanged. It enumerates real reference exposures, never creates an
identity association or silently treats every historical vehicle as co-present.
Coverage, whole-instance, geometry and seam checks must precede final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from model_engine.ogs2_joint_policy import PatchPlan, SparseRegion, regions_overlap


@dataclass(frozen=True)
class ExposureReference:
    capture_time: float
    clock: str


@dataclass(frozen=True)
class ReferenceCohort:
    source_image: str
    capture_time: float
    clock: str
    required_tracks: frozenset[str]
    excluded_temporal_tracks: frozenset[str]


@dataclass(frozen=True)
class ExplicitReferencePlan:
    members: tuple[str, ...]
    support: SparseRegion | None
    historical_tracks: frozenset[str]
    reference_cohorts: tuple[ReferenceCohort, ...]
    rejected_sources: tuple[tuple[str, str], ...]
    rejection_reason: str | None = None


def merge_with_explicit_reference(
    plans: list[PatchPlan],
    exposures: Mapping[str, ExposureReference],
    max_pixels: int = 1_000_000,
) -> list[ExplicitReferencePlan]:
    """Union all historical support, then enumerate actual vehicle-bearing times.

    A complete source catalog is direct observation evidence. Each entry is
    validated against its identity plan's clock and recorded capture interval.
    The selected cohort must still pass ``choose_whole_source``; enumeration
    alone does not certify geometry, completeness, visibility or quality.
    """
    if max_pixels <= 0 or len({p.identity for p in plans}) != len(plans):
        raise ValueError("Require a positive budget and unique patch identities")
    parent = list(range(len(plans)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(plans)):
        for second in range(first + 1, len(plans)):
            if regions_overlap(plans[first].support, plans[second].support):
                parent[find(second)] = find(first)
    components = {}
    for index, plan in enumerate(plans):
        components.setdefault(find(index), []).append(plan)
    result = []
    for component in components.values():
        component.sort(key=lambda p: p.identity)
        members = tuple(p.identity for p in component)
        historical = frozenset(t for p in component for t in p.required_tracks)
        clocks = {p.clock for p in component}
        times = [v for p in component for v in (p.first_time, p.last_time)]
        reason = None
        if not historical:
            reason = "no-required-vehicle"
        elif len(clocks) != 1 or not next(iter(clocks)) or "unknown" in clocks:
            reason = "incompatible-clocks"
        elif not all(math.isfinite(v) for v in times):
            reason = "unknown-capture-time"
        elif any(p.last_time < p.first_time for p in component):
            reason = "invalid-time-interval"
        x0 = min(p.support.bounds[0] for p in component)
        y0 = min(p.support.bounds[1] for p in component)
        x1 = max(p.support.bounds[2] for p in component)
        y1 = max(p.support.bounds[3] for p in component)
        if (x1 - x0) * (y1 - y0) > max_pixels:
            reason = reason or "joint-support-exceeds-budget"
        rejected, cohorts = [], []
        if not reason:
            clock = next(iter(clocks))
            sources = sorted({s for p in component for s in p.complete_tracks_by_source})
            for source in sources:
                reference = exposures.get(source)
                source_reason = None
                if reference is None or not math.isfinite(reference.capture_time):
                    source_reason = "missing-reference-time"
                elif reference.clock != clock:
                    source_reason = "incompatible-reference-clock"
                anchors = set()
                for plan in component:
                    observed = plan.complete_tracks_by_source.get(source, frozenset())
                    if not observed:
                        continue
                    if not observed <= plan.required_tracks:
                        source_reason = source_reason or "source-catalog-has-unplanned-identity"
                    if reference is not None and not (
                        plan.first_time <= reference.capture_time <= plan.last_time
                    ):
                        source_reason = source_reason or "reference-time-outside-observed-track"
                    anchors.update(observed & plan.required_tracks)
                if not anchors:
                    source_reason = source_reason or "no-real-vehicle-anchor"
                if source_reason:
                    rejected.append((source, source_reason))
                else:
                    required = frozenset(anchors)
                    cohorts.append(ReferenceCohort(
                        source, reference.capture_time, clock, required, historical - required
                    ))
            if not cohorts:
                reason = "no-valid-explicit-reference"
        support = None
        if not reason:
            mask = np.zeros((y1 - y0, x1 - x0), bool)
            for plan in component:
                x, y = plan.support.origin
                h, w = plan.support.mask.shape
                mask[y - y0:y - y0 + h, x - x0:x - x0 + w] |= plan.support.mask
            support = SparseRegion((x0, y0), mask)
        result.append(ExplicitReferencePlan(
            members, support, historical, tuple(cohorts), tuple(rejected), reason
        ))
    return sorted(result, key=lambda p: p.members)
