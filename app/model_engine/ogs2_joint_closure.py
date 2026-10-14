"""Order-independent whole-foreground closure before joint patch planning.

Supply only complete masks from the candidate exposure and current output
baseline, not unrelated future exposures. This is geometric reservation, not
vehicle identity association, temporal inference or permission to write pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class WholeForegroundClosure:
    support: np.ndarray
    included_foregrounds: tuple[str, ...]
    passes: int


def _dilate(mask, steps):
    result = mask.copy()
    for _ in range(steps):
        old = result.copy()
        result[1:] |= old[:-1]
        result[:-1] |= old[1:]
        result[:, 1:] |= old[:, :-1]
        result[:, :-1] |= old[:, 1:]
    return result


def close_whole_foregrounds(
    seed_support: np.ndarray,
    foregrounds: Mapping[str, np.ndarray],
    protection_steps: int = 0,
    max_support_pixels: int | None = None,
) -> WholeForegroundClosure:
    """Repeatedly include any actual whole mask touched by the expanded support.

    Masks are stable complete observations. The returned support must undergo
    full source/alpha/geometry/static-collar checks. The budget rejects closure
    growth atomically instead of silently returning a partial foreground.
    """
    seed = np.asarray(seed_support, bool)
    if seed.ndim != 2 or not seed.any():
        raise ValueError("Require a nonempty map-frame support")
    if int(protection_steps) != protection_steps or protection_steps < 0:
        raise ValueError("Require nonnegative integer protection steps")
    if max_support_pixels is not None and max_support_pixels <= 0:
        raise ValueError("Require a positive whole-support pixel budget")
    masks = {key: np.asarray(mask, bool) for key, mask in foregrounds.items()}
    if any(mask.shape != seed.shape for mask in masks.values()):
        raise ValueError("Foreground masks must share the support pixel frame")
    support, included, passes = seed.copy(), set(), 0
    while True:
        passes += 1
        touching = {
            name for name, mask in masks.items()
            if name not in included and (mask & support).any()
        }
        if not touching:
            break
        # Collect all currently touched masks together so dictionary order does
        # not choose which one gets included in this fixed-point pass.
        for name in sorted(touching):
            support |= _dilate(masks[name], int(protection_steps))
        included.update(touching)
        if max_support_pixels is not None and support.sum() > max_support_pixels:
            raise ValueError("Whole-foreground closure exceeds the support budget")
    if max_support_pixels is not None and support.sum() > max_support_pixels:
        raise ValueError("Whole-foreground closure exceeds the support budget")
    if any((mask & support).any() and (mask & ~support).any() for mask in masks.values()):
        raise RuntimeError("Fixed-point closure left a foreground partially cut")
    return WholeForegroundClosure(support, tuple(sorted(included)), passes)


def reserve_joint_components(
    supports: list[np.ndarray],
    candidate_foregrounds: list[Mapping[str, np.ndarray]],
    collar_steps: int,
    protection_steps: int = 0,
    max_support_pixels: int | None = None,
) -> tuple[list[WholeForegroundClosure], list[tuple[int, ...]]]:
    """Close prospective writers, then merge actual reserved collar overlaps.

    Each per-patch catalog is the union of foregrounds relevant to its real
    candidate exposures/current baseline. Reservation may exceed final writing
    support; it never authorizes a mixed-time vehicle composite.
    """
    if len(supports) != len(candidate_foregrounds):
        raise ValueError("Each patch requires its actual candidate foreground catalog")
    if int(collar_steps) != collar_steps or collar_steps < 0:
        raise ValueError("Require a nonnegative integer reservation collar")
    closures = [close_whole_foregrounds(seed, masks, protection_steps, max_support_pixels)
                for seed, masks in zip(supports, candidate_foregrounds)]
    if not closures:
        return [], []
    if any(c.support.shape != closures[0].support.shape for c in closures):
        raise ValueError("Every patch must use the same map pixel frame")
    parent = list(range(len(closures)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    while True:
        reserved = [_dilate(c.support, int(collar_steps)) for c in closures]
        changed = False
        for first in range(len(reserved)):
            for second in range(first + 1, len(reserved)):
                if find(first) != find(second) and (reserved[first] & reserved[second]).any():
                    parent[find(second)] = find(first)
                    changed = True
        grouped = {}
        for index in range(len(reserved)):
            grouped.setdefault(find(index), []).append(index)
        for indices in grouped.values():
            seed = np.logical_or.reduce([closures[index].support for index in indices])
            catalog = {}
            for index in indices:
                for name, mask in candidate_foregrounds[index].items():
                    if name in catalog and not np.array_equal(catalog[name], mask):
                        raise ValueError("One foreground observation has contradictory masks")
                    catalog[name] = mask
            joint = close_whole_foregrounds(seed, catalog, protection_steps, max_support_pixels)
            for index in indices:
                if not np.array_equal(closures[index].support, joint.support):
                    changed = True
                closures[index] = joint
        # A newly merged component can encounter a foreground in another
        # member's candidate catalog, then expand to another writer. Recompute
        # both closure and components until neither can change.
        if not changed:
            return closures, sorted(tuple(indices) for indices in grouped.values())
