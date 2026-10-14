"""Experimental closed hard patches between two already aligned RGB rasters.

The cut prefers similar RGB and gradients. These quantities do not establish
correct road geometry, an empty road, or semantic correctness. No image is
warped, blended, relit, or invented. Mask provenance remains the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow


@dataclass(frozen=True)
class RoadPatchConfig:
    max_pixels: int = 2_000_000
    capacity_levels: int = 256
    rgb_weight: float = 1.0
    gradient_weight: float = 1.0


@dataclass(frozen=True)
class RoadPatchResult:
    accepted: bool
    reason: str
    take_candidate: np.ndarray | None
    source_index: np.ndarray | None
    seam_mask: np.ndarray | None
    qc: dict
    bindings: dict


def _digest(value):
    value = np.asarray(value)
    result = hashlib.sha256()
    result.update(str(value.dtype).encode())
    result.update(str(value.shape).encode())
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask on the common grid")
    return value


def _images(base, candidate):
    base, candidate = np.asarray(base), np.asarray(candidate)
    if (base.ndim != 3 or base.shape[-1] != 3 or base.dtype != np.uint8
            or candidate.shape != base.shape or candidate.dtype != np.uint8
            or min(base.shape[:2]) < 3):
        raise ValueError("Require two uint8 HxWx3 RGB rasters with dimensions >= 3")
    return base, candidate


def select_road_patch(
    base, candidate, valid_base, valid_candidate, hard_take_candidate,
    allowed_candidate, forbidden_cut, *, config=RoadPatchConfig(),
):
    """Compute an experimental 4-neighbour closed patch, or explicitly refuse.

    A selected candidate pixel is label 1; unchanged base is label 0. A one-pixel
    image border must remain base, so the patch is closed within this raster.
    Every actual cut edge, including stair turns, must avoid ``forbidden_cut``
    at BOTH endpoints. Invalid source pixels are hard constraints, not fallback
    opportunities. The complete output must have valid source coverage.

    ``accepted`` means only a feasible discrete optimization, not acceptance of
    scene geometry, vehicle masks, output quality, or a final orthophoto.
    """
    base, candidate = _images(base, candidate)
    shape = base.shape[:2]
    valid_base = _mask(valid_base, shape, "valid_base")
    valid_candidate = _mask(valid_candidate, shape, "valid_candidate")
    hard = _mask(hard_take_candidate, shape, "hard_take_candidate")
    allowed = _mask(allowed_candidate, shape, "allowed_candidate")
    forbidden = _mask(forbidden_cut, shape, "forbidden_cut")
    if (isinstance(config.max_pixels, bool) or not isinstance(config.max_pixels, int)
            or not 9 <= config.max_pixels <= 2_000_000):
        raise ValueError("max_pixels must be an integer in [9, 2000000]")
    if (isinstance(config.capacity_levels, bool) or not isinstance(config.capacity_levels, int)
            or not 2 <= config.capacity_levels <= 256):
        raise ValueError("capacity_levels must be an integer in [2, 256]")
    weights = np.asarray([config.rgb_weight, config.gradient_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.max() <= 0:
        raise ValueError("RGB/gradient weights must be finite, nonnegative, and nonzero together")
    # Normalization before summation avoids overflow for large finite weights.
    weights /= weights.max()
    weights /= weights.sum()
    qc = {
        "experimentalCandidate": True, "geometryCertified": False,
        "semanticsCertified": False, "wholeOrthophotoCertified": False,
        "mixedPixels": 0, "sourceIndexConvention": {"0": "base", "1": "candidate"},
        "connectivity": 4, "capacityLevels": config.capacity_levels,
    }

    def refuse(reason):
        return RoadPatchResult(False, reason, None, None, None, dict(qc), {})

    if not hard.any():
        return refuse("hard_candidate_core_is_empty")
    if (hard & ~allowed).any():
        return refuse("hard_candidate_outside_allowed_region")
    if (hard & ~valid_candidate).any():
        return refuse("hard_candidate_has_invalid_source")
    if (~valid_base & ~valid_candidate).any():
        return refuse("both_sources_invalid")
    if (~valid_base & ~allowed).any():
        return refuse("base_invalid_outside_allowed_region")

    # Crop with an explicit fixed-base collar. No neighbouring outside node can
    # influence the cut because every crop perimeter node is fixed to base.
    yy, xx = np.nonzero(allowed)
    y0, y1 = max(0, int(yy.min()) - 1), min(shape[0], int(yy.max()) + 2)
    x0, x1 = max(0, int(xx.min()) - 1), min(shape[1], int(xx.max()) + 2)
    sl = np.s_[y0:y1, x0:x1]
    h, w = y1 - y0, x1 - x0
    n = h * w
    qc.update(cropXYXY=[x0, y0, x1, y1], graphPixels=n)
    if n > config.max_pixels:
        return refuse("graph_pixel_budget_exceeded")
    local_allowed = allowed[sl]
    fixed_base = (~local_allowed | ~valid_candidate[sl]).copy()
    fixed_base[[0, -1], :] = True
    fixed_base[:, [0, -1]] = True
    fixed_candidate = hard[sl] | ~valid_base[sl]
    if (fixed_base & fixed_candidate).any():
        return refuse("conflicting_hard_constraints_or_nonclosed_patch")

    ids = np.arange(n, dtype=np.int32).reshape(h, w)
    p = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
    q = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
    f = forbidden[sl].ravel()
    protected_edge = f[p] | f[q]

    # Every protected component must have one source. Check feasibility first,
    # preventing an impossible graph from routing multiple huge terminal flows.
    if protected_edge.any():
        fp, fq = p[protected_edge], q[protected_edge]
        protected_graph = coo_matrix(
            (np.ones(len(fp) * 2, np.uint8),
             (np.concatenate((fp, fq)), np.concatenate((fq, fp)))), shape=(n, n),
        ).tocsr()
        count, component = connected_components(protected_graph, directed=False)
        has_base = np.zeros(count, bool)
        has_candidate = np.zeros(count, bool)
        has_base[component[fixed_base.ravel()]] = True
        has_candidate[component[fixed_candidate.ravel()]] = True
        if (has_base & has_candidate).any():
            return refuse("forbidden_cut_component_conflicts_with_hard_constraints")
        del protected_graph, component, has_base, has_candidate

    a = base[sl].reshape(-1, 3).astype(np.float32)
    b = candidate[sl].reshape(-1, 3).astype(np.float32)
    pixel_difference = np.abs(a - b).mean(axis=1)
    rgb_cost = (pixel_difference[p] + pixel_difference[q]) / (2.0 * 255.0)
    gradient_cost = np.abs(np.abs(a[p] - a[q]) - np.abs(b[p] - b[q])).mean(axis=1) / 255.0
    normalized = np.clip(weights[0] * rgb_cost + weights[1] * gradient_cost, 0.0, 1.0)
    finite_cost = 1 + np.rint(normalized * (config.capacity_levels - 1)).astype(np.int64)
    finite_total = int(finite_cost[~protected_edge].sum(dtype=np.int64))
    infinity = finite_total + 1
    # scipy maximum_flow has used signed-32-bit internal flows across versions.
    # The feasible minimum cut is <= finite_total, and each edge is <= infinity.
    # The configured pixel and quantization limits enforce this conservative cap.
    if infinity > np.iinfo(np.int32).max:
        return refuse("integer_capacity_budget_exceeded")
    costs = np.where(protected_edge, infinity, finite_cost)
    source, sink = n, n + 1
    hard_c = np.flatnonzero(fixed_candidate.ravel())
    hard_b = np.flatnonzero(fixed_base.ravel())
    rows = np.concatenate((p, q, np.full(hard_c.size, source), hard_b))
    cols = np.concatenate((q, p, hard_c, np.full(hard_b.size, sink)))
    capacities = np.concatenate((costs, costs,
                                 np.full(hard_c.size + hard_b.size, infinity, dtype=np.int64)))
    graph = coo_matrix((capacities, (rows, cols)), shape=(n + 2, n + 2), dtype=np.int64).tocsr()
    flow = maximum_flow(graph, source, sink, method="dinic")
    residual = graph - flow.flow.astype(np.int64)
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    local_take = np.zeros(n, bool)
    local_take[reachable[reachable < n]] = True
    edge_cut = local_take[p] != local_take[q]
    flow_value = int(flow.flow_value)
    qc.update(flowValue=flow_value, finiteCapacitySum=finite_total, infiniteCapacity=infinity,
              cutEdges=int(edge_cut.sum()))
    if flow_value >= infinity or (edge_cut & protected_edge).any():
        return refuse("forbidden_edge_in_actual_cut")
    local_take = local_take.reshape(h, w)
    if (fixed_base & local_take).any() or (fixed_candidate & ~local_take).any():
        return refuse("actual_cut_violates_hard_constraint")
    take = np.zeros(shape, bool)
    take[sl] = local_take
    if ((take & (~allowed | ~valid_candidate)).any()
            or (~take & ~valid_base).any() or (hard & ~take).any()):
        return refuse("actual_output_has_invalid_coverage_or_constraint_violation")
    seam_local = np.zeros(n, bool)
    seam_local[p[edge_cut]] = True
    seam_local[q[edge_cut]] = True
    seam = np.zeros(shape, bool)
    seam[sl] = seam_local.reshape(h, w)
    if (seam & forbidden).any():
        return refuse("forbidden_endpoint_in_actual_cut")
    source_index = take.astype(np.uint8)
    selected_cost = finite_cost[edge_cut]
    qc.update(candidatePixels=int(take.sum()), seamEndpointPixels=int(seam.sum()),
              cutCapacity=int(selected_cost.sum(dtype=np.int64)),
              seamRgbDifferenceMean=float(pixel_difference[seam_local].mean()),
              seamRgbDifferenceMax=float(pixel_difference[seam_local].max()),
              invalidSelectedPixels=0, forbiddenCutEndpoints=0)
    if qc["cutCapacity"] != flow_value:
        return refuse("flow_and_actual_cut_cost_disagree")
    bindings = {"base": _digest(base), "candidate": _digest(candidate),
                "takeCandidate": _digest(take), "sourceIndex": _digest(source_index),
                "seamMask": _digest(seam), "validBase": _digest(valid_base),
                "validCandidate": _digest(valid_candidate), "hardCandidate": _digest(hard),
                "allowedCandidate": _digest(allowed), "forbiddenCut": _digest(forbidden)}
    return RoadPatchResult(True, "feasible_experimental_hard_patch", take, source_index, seam, qc, bindings)


def compose_road_patch(base, candidate, result):
    """Copy exactly one source per pixel after checking the solved-plan binding."""
    base, candidate = _images(base, candidate)
    if not result.accepted or result.take_candidate is None:
        raise ValueError("Cannot compose a refused patch")
    checks = {"base": base, "candidate": candidate, "takeCandidate": result.take_candidate,
              "sourceIndex": result.source_index, "seamMask": result.seam_mask}
    if any(result.bindings.get(name) != _digest(value) for name, value in checks.items()):
        raise ValueError("Image or solved plan changed after cut selection")
    output = base.copy()
    output[result.take_candidate] = candidate[result.take_candidate]
    return output, result.source_index.copy(), dict(result.qc)
