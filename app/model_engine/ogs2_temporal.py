"""Conservative, local, short-time vehicle association for OGS2.

This module proposes conflict regions; it does not identify every vehicle in a
scene or erase observations. Masks stay in local windows with global origins.
The caller must verify a single source covers a whole conflict region before
using it, and must not fill an unsupported region with invented background.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TemporalConfig:
    max_gap_seconds: float = 3.0
    min_confidence: float = 0.8
    max_speed_mps: float = 25.0
    static_tolerance_m: float = 0.25
    max_area_ratio: float = 1.8
    max_appearance_distance: float = 0.12
    best_margin: float = 0.08
    duplicate_iou: float = 0.6
    duplicate_containment: float = 0.92
    max_group_observations: int = 12
    max_group_span_seconds: float = 3.0
    max_group_diameter_m: float = 80.0
    max_conflict_pixels: int = 4_000_000
    max_envelope_to_core_ratio: float = 12.0
    require_explicit_clock: bool = False


@dataclass
class _Observation:
    index: int
    view: int
    capture_time: float | None
    clock: str | None
    mask: np.ndarray
    origin: tuple[int, int]
    centroid: np.ndarray
    area: int
    complete: bool
    confidence: float
    appearance: np.ndarray | None
    category: str

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        x, y = self.origin
        return x, y, x + self.mask.shape[1], y + self.mask.shape[0]


def _parse(index: int, observation: dict[str, Any]) -> _Observation:
    mask = np.asarray(observation["mask"], dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError(f"Observation {index} needs a nonempty 2D mask")
    origin = observation.get("origin", (0, 0))
    if len(origin) != 2 or any(not math.isfinite(float(value)) for value in origin):
        raise ValueError(f"Observation {index} has an invalid origin")
    if any(float(value) != int(value) for value in origin):
        raise ValueError("Mask origins must use integer global pixel coordinates")
    origin = int(origin[0]), int(origin[1])
    ys, xs = np.nonzero(mask)
    # Crop padded input masks before pairwise overlap work.
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tight = mask[top:bottom, left:right]
    tight_origin = origin[0] + int(left), origin[1] + int(top)
    centroid = np.array([xs.mean() + origin[0], ys.mean() + origin[1]])
    appearance = observation.get("appearance")
    if appearance is not None:
        appearance = np.asarray(appearance, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(appearance)
        if not appearance.size or not np.isfinite(appearance).all() or norm <= 0:
            appearance = None
        else:
            appearance = appearance / norm
    capture_time = observation.get("capture_time")
    if capture_time is not None:
        capture_time = float(capture_time)
        if not math.isfinite(capture_time):
            capture_time = None
    # Epoch-looking numbers do not prove that a timezone-less DJI filename and
    # an OpenSfM timestamp share a clock. Domain equality is always required.
    if "clock" not in observation:
        clock = "legacy-test-clock"
    else:
        value = observation["clock"]
        clock = value.strip().lower() if isinstance(value, str) else None
        if clock in {"", "unknown", "none", "null", "nan"}:
            clock = None
    return _Observation(
        index=index,
        view=int(observation["view"]),
        capture_time=capture_time,
        clock=clock,
        mask=tight,
        origin=tight_origin,
        centroid=centroid,
        area=int(xs.size),
        complete=bool(observation.get("complete", False)),
        confidence=float(observation.get("confidence", 0.0)),
        appearance=appearance,
        category=str(observation.get("category", "")).strip().lower(),
    )


def _intersection(a: _Observation, b: _Observation) -> int:
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0
    am = a.mask[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
    bm = b.mask[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
    return int(np.count_nonzero(am & bm))


def _is_duplicate(a: _Observation, b: _Observation, config: TemporalConfig) -> bool:
    if a.view != b.view or a.clock != b.clock:
        return False
    intersection = _intersection(a, b)
    iou = intersection / (a.area + b.area - intersection)
    if iou > config.duplicate_iou:
        return True
    # Allow cropped copies only when they still cover a substantial proportion
    # of each other. A tiny core inside a large leaked mask is not a duplicate.
    ratio = min(a.area, b.area) / max(a.area, b.area)
    containment = intersection / min(a.area, b.area)
    return containment >= config.duplicate_containment and ratio >= 0.65


def _deduplicate(
    observations: list[_Observation], config: TemporalConfig
) -> tuple[list[_Observation], list[dict[str, int]]]:
    ordered = sorted(
        observations,
        key=lambda item: (
            item.complete,
            item.confidence if math.isfinite(item.confidence) else -math.inf,
            item.capture_time is not None,
            item.area,
            -item.index,
        ),
        reverse=True,
    )
    retained: list[_Observation] = []
    duplicates: list[dict[str, int]] = []
    by_view: dict[int, list[_Observation]] = {}
    for item in ordered:
        representative = next(
            (other for other in by_view.get(item.view, []) if _is_duplicate(item, other, config)),
            None,
        )
        if representative is not None:
            duplicates.append({"discarded": item.index, "retained": representative.index})
        else:
            retained.append(item)
            by_view.setdefault(item.view, []).append(item)
    return sorted(retained, key=lambda item: item.index), duplicates


def _appearance_distance(a: _Observation, b: _Observation) -> float | None:
    if a.appearance is None or b.appearance is None or a.appearance.shape != b.appearance.shape:
        return None
    return max(0.0, float(1.0 - np.dot(a.appearance, b.appearance)))


def _pair_score(
    a: _Observation, b: _Observation, pixel_size_m: float, config: TemporalConfig
) -> tuple[float | None, str, float]:
    if a.view == b.view:
        return None, "same_view", 0.0
    if a.clock is None or a.clock != b.clock:
        return None, "clock_domain", 0.0
    assert a.capture_time is not None and b.capture_time is not None
    dt = b.capture_time - a.capture_time
    if dt <= 0 or dt > config.max_gap_seconds:
        return None, "time_gap", 0.0
    displacement = float(np.linalg.norm(a.centroid - b.centroid) * pixel_size_m)
    if displacement > config.max_speed_mps * dt + config.static_tolerance_m:
        return None, "speed", displacement
    if a.category and b.category and a.category != b.category:
        return None, "category", displacement
    area_ratio = max(a.area, b.area) / min(a.area, b.area)
    if area_ratio > config.max_area_ratio:
        return None, "area", displacement
    appearance = _appearance_distance(a, b)
    if appearance is None or appearance > config.max_appearance_distance:
        return None, "appearance", displacement
    # Static candidates participate in the competition; excluding them would
    # allow a parked car to attach to a nearby moving lookalike instead.
    score = (
        4.0 * appearance
        + 0.4 * math.log(area_ratio)
        + 0.15 * displacement / (config.max_speed_mps * dt + config.static_tolerance_m)
        + 0.03 * dt / config.max_gap_seconds
    )
    return score, "candidate", displacement


def _clear_best(candidates: list[tuple[float, int]], margin: float) -> int | None:
    candidates.sort()
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < margin:
        return None
    return candidates[0][1]


def associate_observations(
    observations: list[dict[str, Any]],
    pixel_size_m: float,
    config: TemporalConfig | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return conservative dynamic groups using original observation indices.

    Association uses positive capture-time differences within one clock domain.
    Explicit unknown domains are withheld. Missing clock fields are allowed only
    for legacy synthetic inputs lacking sourceImage and when strict mode is off.
    Production should always use require_explicit_clock=True. Returned groups
    are local hypotheses with unique views, not globally guaranteed identities.
    """
    config = config or TemporalConfig()
    if not math.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive and finite")
    if config.max_gap_seconds <= 0 or config.max_speed_mps <= 0:
        raise ValueError("Time and speed limits must be positive")
    parsed = [_parse(index, item) for index, item in enumerate(observations)]
    retained, duplicates = _deduplicate(parsed, config)
    rejected = Counter()
    eligible = []
    for item in retained:
        if item.capture_time is None:
            rejected["missing_time"] += 1
        elif item.clock is None:
            rejected["unknown_clock"] += 1
        elif item.clock == "legacy-test-clock" and (
            config.require_explicit_clock or "sourceImage" in observations[item.index]
        ):
            rejected["missing_clock"] += 1
        elif not item.complete:
            rejected["incomplete"] += 1
        elif not math.isfinite(item.confidence) or item.confidence < config.min_confidence:
            rejected["confidence"] += 1
        elif item.appearance is None:
            rejected["missing_appearance"] += 1
        else:
            eligible.append(item)
    # The sorted short-time sweep stays within a single clock domain. Numeric
    # proximity in another clock cannot compete for or hide a valid match.
    eligible.sort(key=lambda item: (item.clock, item.capture_time, item.index))
    forward: dict[int, list[tuple[float, int]]] = {item.index: [] for item in eligible}
    backward: dict[int, list[tuple[float, int]]] = {item.index: [] for item in eligible}
    pair_displacements: dict[tuple[int, int], float] = {}
    candidate_count = 0
    for position, item in enumerate(eligible):
        for other_position in range(position + 1, len(eligible)):
            other = eligible[other_position]
            assert item.capture_time is not None and other.capture_time is not None
            if other.clock != item.clock:
                break
            if other.capture_time - item.capture_time > config.max_gap_seconds:
                break
            score, reason, displacement = _pair_score(item, other, pixel_size_m, config)
            if score is None:
                rejected[reason] += 1
                continue
            candidate_count += 1
            forward[item.index].append((score, other.index))
            backward[other.index].append((score, item.index))
            pair_displacements[item.index, other.index] = displacement
    by_index = {item.index: item for item in eligible}
    # Compare identities at the nearest observed time in each direction. The
    # same physical car at t+1 and t+2 is not an identity ambiguity. Ambiguity
    # between neighbours within the same target capture remains a hard veto.
    for mapping, choose_time in ((forward, min), (backward, max)):
        for index, candidates in mapping.items():
            if candidates:
                nearest_time = choose_time(by_index[other].capture_time for _, other in candidates)
                mapping[index] = [
                    pair for pair in candidates if by_index[pair[1]].capture_time == nearest_time
                ]
    forward_best = {key: _clear_best(value, config.best_margin) for key, value in forward.items()}
    backward_best = {key: _clear_best(value, config.best_margin) for key, value in backward.items()}
    rejected["ambiguous_direction"] = sum(
        bool(value) and best is None
        for mapping, choices in ((forward, forward_best), (backward, backward_best))
        for key, value in mapping.items()
        for best in [choices[key]]
    )
    edges = []
    for first, second in forward_best.items():
        if second is not None and backward_best.get(second) == first:
            edges.append((first, second))
        elif second is not None:
            rejected["non_mutual"] += 1
    groups: list[set[int]] = [{item.index} for item in eligible]
    membership = {item.index: position for position, item in enumerate(eligible)}
    accepted_edges: list[tuple[int, int]] = []
    for first, second in edges:
        first_group, second_group = membership[first], membership[second]
        if first_group == second_group:
            continue
        combined = groups[first_group] | groups[second_group]
        members = [by_index[index] for index in sorted(combined)]
        views = [item.view for item in members]
        times = [float(item.capture_time) for item in members]
        if len(set(views)) != len(views):
            rejected["group_repeated_view"] += 1
            continue
        if len(members) > config.max_group_observations:
            rejected["group_size"] += 1
            continue
        if max(times) - min(times) > config.max_group_span_seconds:
            rejected["group_span"] += 1
            continue
        centroids = np.array([item.centroid for item in members]) * pixel_size_m
        diameter = float(np.max(np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)))
        if diameter > config.max_group_diameter_m:
            rejected["group_diameter"] += 1
            continue
        distances = [_appearance_distance(a, b) for a in members for b in members]
        if any(value is None or value > config.max_appearance_distance for value in distances):
            rejected["group_appearance_drift"] += 1
            continue
        width = max(item.bounds[2] for item in members) - min(item.bounds[0] for item in members)
        height = max(item.bounds[3] for item in members) - min(item.bounds[1] for item in members)
        envelope_pixels = width * height
        if envelope_pixels > config.max_conflict_pixels:
            rejected["group_conflict_size"] += 1
            continue
        core_area = sum(item.area for item in members)
        if envelope_pixels > config.max_envelope_to_core_ratio * core_area:
            rejected["group_sparse_envelope"] += 1
            continue
        groups[first_group] = combined
        groups[second_group] = set()
        for index in combined:
            membership[index] = first_group
        accepted_edges.append((first, second))
    result = []
    summaries = []
    for group in groups:
        if len(group) < 2:
            continue
        group_edges = [(a, b) for a, b in accepted_edges if a in group and b in group]
        movement = max(pair_displacements[a, b] for a, b in group_edges)
        if movement <= config.static_tolerance_m:
            rejected["static_group"] += 1
            continue
        ordered = sorted(group, key=lambda index: (by_index[index].capture_time, index))
        result.append(ordered)
        summaries.append(
            {
                "indices": ordered,
                "dynamic_confirmed": True,
                "clock": by_index[ordered[0]].clock,
                "maximum_step_m": movement,
                "span_seconds": float(by_index[ordered[-1]].capture_time)
                - float(by_index[ordered[0]].capture_time),
            }
        )
    diagnostics: dict[str, Any] = {
        "input_observations": len(observations),
        "retained_observations": len(retained),
        "eligible_observations": len(eligible),
        "clock_domains": dict(Counter(item.clock for item in eligible)),
        "duplicates": duplicates,
        "candidate_pairs": candidate_count,
        "accepted_edges": [list(edge) for edge in accepted_edges],
        "rejected": dict(rejected),
        "groups": summaries,
        "identity_scope": "conservative local short-time hypotheses; no global identity guarantee",
    }
    return result, diagnostics


def render_conflict(
    group: list[int],
    observations: list[dict[str, Any]],
    padding_pixels: int = 3,
    max_pixels: int = 4_000_000,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Render a bounded local union, a short swept connector, and a small buffer.

    The connector follows observed centroids; it is a conflict exclusion region,
    not a newly reconstructed vehicle surface. Huge envelopes fail explicitly.
    """
    if not group or padding_pixels < 0 or int(padding_pixels) != padding_pixels:
        raise ValueError("A nonempty group and nonnegative integer padding are required")
    members = [_parse(index, observations[index]) for index in group]
    padding = int(padding_pixels)
    x0 = min(item.bounds[0] for item in members) - padding
    y0 = min(item.bounds[1] for item in members) - padding
    x1 = max(item.bounds[2] for item in members) + padding
    y1 = max(item.bounds[3] for item in members) + padding
    if (x1 - x0) * (y1 - y0) > max_pixels:
        raise ValueError("Conflict envelope exceeds the bounded local pixel budget")
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for item in members:
        x, y = item.origin[0] - x0, item.origin[1] - y0
        mask[y : y + item.mask.shape[0], x : x + item.mask.shape[1]] |= item.mask
    ordered = sorted(
        members,
        key=lambda item: (item.capture_time is None, item.capture_time or 0.0, item.index),
    )
    for first, second in zip(ordered, ordered[1:]):
        start, end = first.centroid - [x0, y0], second.centroid - [x0, y0]
        steps = max(2, int(np.ceil(np.max(np.abs(end - start)))) + 1)
        coordinates = np.rint(np.linspace(start, end, steps)).astype(int)
        connector = np.zeros_like(mask)
        connector[coordinates[:, 1], coordinates[:, 0]] = True
        # A bounded capsule with a conservative half-width derived from observed
        # footprint widths covers gaps between the vehicle's observed positions.
        radius = max(1, int(math.ceil(min(*first.mask.shape, *second.mask.shape) / 2)))
        # An EDT capsule avoids very large dilation kernels on large masks.
        mask |= ndimage.distance_transform_edt(~connector) <= radius
    if padding:
        yy, xx = np.ogrid[-padding : padding + 1, -padding : padding + 1]
        disk = xx * xx + yy * yy <= padding * padding
        mask = ndimage.binary_dilation(mask, structure=disk)
    return mask, (x0, y0)
