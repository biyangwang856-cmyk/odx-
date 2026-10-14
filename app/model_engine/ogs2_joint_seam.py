"""Constrained, unwarped two-source seams around a single-photo dynamic core.

Production uses the existing SciPy maximum-flow backend. The bounded Python
reference backend exists for tiny CPU contract tests only, never full imagery.
No image, camera, vehicle or shadow is geometrically moved by this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeamConfig:
    max_pixels: int = 350_000
    min_hard_margin_px: int = 2
    feather_radius_px: int = 2
    feather_max_rgb_difference: float = 12.0
    feather_max_gradient: float = 10.0
    python_max_pixels: int = 4096


def _dilate(mask, steps):
    result = mask.copy()
    for _ in range(steps):
        old = result.copy()
        result[1:] |= old[:-1]
        result[:-1] |= old[1:]
        result[:, 1:] |= old[:, :-1]
        result[:, :-1] |= old[:, 1:]
    return result


def _gradient(image):
    image = image.astype(np.float32)
    out = np.zeros(image.shape[:2], np.float32)
    for axis in (0, 1):
        change = np.abs(np.diff(image, axis=axis)).mean(axis=2)
        if axis == 0:
            out[:-1] = np.maximum(out[:-1], change)
            out[1:] = np.maximum(out[1:], change)
        else:
            out[:, :-1] = np.maximum(out[:, :-1], change)
            out[:, 1:] = np.maximum(out[:, 1:], change)
    return out


def _python_cut(rows, columns, capacities, nodes, source, sink):
    graph = [[] for _ in range(nodes)]
    for first, second, capacity in zip(rows, columns, capacities):
        first, second, capacity = int(first), int(second), int(capacity)
        graph[first].append([second, len(graph[second]), capacity])
        graph[second].append([first, len(graph[first]) - 1, 0])
    flow = 0
    while True:
        previous = [None] * nodes
        previous[source] = (-1, -1)
        queue = deque([source])
        while queue and previous[sink] is None:
            first = queue.popleft()
            for edge_index, (second, _, capacity) in enumerate(graph[first]):
                if capacity > 0 and previous[second] is None:
                    previous[second] = (first, edge_index)
                    queue.append(second)
        if previous[sink] is None:
            reachable = np.asarray([item is not None for item in previous])
            return reachable, flow
        node, amount = sink, None
        while node != source:
            first, edge_index = previous[node]
            capacity = graph[first][edge_index][2]
            amount = capacity if amount is None else min(amount, capacity)
            node = first
        node = sink
        while node != source:
            first, edge_index = previous[node]
            second, reverse, _ = graph[first][edge_index]
            graph[first][edge_index][2] -= amount
            graph[second][reverse][2] += amount
            node = first
        flow += amount


def _scipy_cut(rows, columns, capacities, nodes, source, sink):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    # SciPy 1.14.1 casts capacities to its compiled ITYPE. Bound every capacity
    # and the known feasible cut below int32 before construction, not afterward.
    graph = coo_matrix((capacities, (rows, columns)), shape=(nodes, nodes)).tocsr()
    result = maximum_flow(graph, source, sink, method="dinic")
    residual = graph - result.flow
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    indices = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    reachable = np.zeros(nodes, bool)
    reachable[indices] = True
    return reachable, int(result.flow_value)


def _distance_to_false(mask, backend):
    if backend == "scipy":
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(mask).astype(np.float32)
    # A Manhattan-distance reference is sufficient to test mask/provenance
    # constraints. Production uses Euclidean distances, not this slow path.
    height, width = mask.shape
    result = np.full(mask.shape, height + width + 1, np.float32)
    queue = deque()
    for row, column in zip(*np.where(~mask)):
        result[row, column] = 0
        queue.append((int(row), int(column)))
    while queue:
        row, column = queue.popleft()
        for nr, nc in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= nr < height and 0 <= nc < width and result[nr, nc] > result[row, column] + 1:
                result[nr, nc] = result[row, column] + 1
                queue.append((nr, nc))
    return result


def static_seam_weights(
    baseline_rgba,
    source_rgb,
    hard_support,
    verified_static,
    valid_source,
    protected_foregrounds=(),
    config=None,
    backend="scipy",
    collar_pixels=None,
):
    """Return a hard-constrained cut, limited static feather weights and audit.

    `hard_support` includes the full vehicle/shadow/dynamic-difference union.
    Every additional foreground mask is preserved whole: intersecting the hard
    support partially rejects, while an unrelated whole mask stays baseline.
    An unknown pixel can never become source or blended content.
    """
    config = config or SeamConfig()
    baseline, source = np.asarray(baseline_rgba), np.asarray(source_rgb)
    if baseline.dtype != np.uint8 or source.dtype != np.uint8:
        raise ValueError("Require explicit uint8 source and baseline")
    if baseline.ndim != 3 or baseline.shape[2] != 4 or source.shape != (*baseline.shape[:2], 3):
        raise ValueError("Require baseline HxWx4 and source HxWx3 in one pixel frame")
    shape = baseline.shape[:2]
    hard, static, valid = [np.asarray(value, bool) for value in (hard_support, verified_static, valid_source)]
    if any(value.shape != shape for value in (hard, static, valid)) or not hard.any():
        raise ValueError("Require matched, nonempty hard support")
    if backend not in ("scipy", "python"):
        raise ValueError("Backend must be scipy or bounded python reference")
    if config.min_hard_margin_px < 0 or config.feather_radius_px < 0:
        raise ValueError("Margin and feather radius cannot be negative")
    pixels = shape[0] * shape[1]
    if pixels > config.max_pixels or (backend == "python" and pixels > config.python_max_pixels):
        raise ValueError("Seam exceeds the bounded pixel budget")
    covered = baseline[..., 3] > 0
    if (hard & (~valid | ~covered)).any():
        raise ValueError("Dynamic hard support has unknown source or baseline coverage")
    boundary = np.zeros(shape, bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    if (hard & boundary).any():
        raise ValueError("Expand context; hard support touches the window boundary")
    foreground = np.zeros(shape, bool)
    for raw in protected_foregrounds:
        mask = np.asarray(raw, bool)
        if mask.shape != shape:
            raise ValueError("Foreground mask uses another pixel frame")
        intersection = int((mask & hard).sum())
        if 0 < intersection < int(mask.sum()):
            raise ValueError("Hard support cuts a protected foreground; merge or expand it")
        if not intersection:
            foreground |= mask
    allowed_static = static & valid & covered & ~foreground
    if collar_pixels is not None:
        if int(collar_pixels) != collar_pixels or collar_pixels < config.min_hard_margin_px:
            raise ValueError("Static collar must contain the configured hard margin")
        allowed_static &= _distance_to_false(~hard, backend) <= int(collar_pixels)
    source_seeds = _dilate(hard, config.min_hard_margin_px)
    if (source_seeds & ~hard & ~allowed_static).any() or (source_seeds & boundary).any():
        raise ValueError("No verified static collar surrounds the complete hard support")
    force_baseline = (~hard & ~allowed_static) | boundary
    mismatch = np.abs(baseline[..., :3].astype(np.float32) - source.astype(np.float32)).mean(axis=2)
    gradient = np.maximum(_gradient(baseline[..., :3]), _gradient(source))
    node_map = np.arange(pixels, dtype=np.int32).reshape(shape)
    first = np.concatenate([node_map[:, :-1].ravel(), node_map[:-1, :].ravel()])
    second = np.concatenate([node_map[:, 1:].ravel(), node_map[1:, :].ravel()])
    local_cost = mismatch + gradient
    pair_cost = 1 + np.rint(local_cost.ravel()[first] + local_cost.ravel()[second]).astype(np.int64)
    # Using source_seeds alone is a feasible cut. Its energy is a smaller and
    # safe infinity bound than summing every edge of a large image graph.
    crosses_feasible = source_seeds.ravel()[first] != source_seeds.ravel()[second]
    feasible_energy = int(pair_cost[crosses_feasible].sum())
    infinity = feasible_energy + 1
    if infinity >= np.iinfo(np.int32).max:
        raise ValueError("Integer flow capacity budget exceeded; split the verified context")
    source_node, sink_node = pixels, pixels + 1
    source_ids = node_map[source_seeds]
    baseline_ids = node_map[force_baseline]
    rows = np.concatenate([first, second, np.full(len(source_ids), source_node), baseline_ids])
    columns = np.concatenate([second, first, source_ids, np.full(len(baseline_ids), sink_node)])
    capacities = np.concatenate([
        pair_cost, pair_cost, np.full(len(source_ids), infinity), np.full(len(baseline_ids), infinity),
    ]).astype(np.int64)
    solver = _scipy_cut if backend == "scipy" else _python_cut
    reachable, flow_value = solver(rows, columns, capacities, pixels + 2, source_node, sink_node)
    selected = reachable[:pixels].reshape(shape)
    if not selected[source_seeds].all() or (selected & force_baseline).any():
        raise RuntimeError("Minimum cut violated a hard terminal constraint")
    weight = selected.astype(np.float32)
    feather_allowed = allowed_static & (mismatch <= config.feather_max_rgb_difference)
    feather_allowed &= gradient <= config.feather_max_gradient
    if config.feather_radius_px:
        signed = _distance_to_false(selected, backend) - _distance_to_false(~selected, backend)
        soft = np.clip(0.5 + signed / (2 * config.feather_radius_px), 0, 1)
        feather = feather_allowed & ~source_seeds & ~force_baseline
        weight[feather] = soft[feather]
    weight[source_seeds] = 1
    weight[force_baseline] = 0
    mixed = (weight > 0) & (weight < 1)
    if not (weight[hard] == 1).all() or (mixed & ~feather_allowed).any():
        raise RuntimeError("Foreground/unknown pixels entered a mixed seam")
    if ((weight > 0) & ~hard & ~allowed_static).any():
        raise RuntimeError("Unknown content became source pixels")
    mode = np.zeros(shape, np.uint8)
    mode[weight == 1] = 1
    mode[mixed] = 2
    crosses = selected.ravel()[first] != selected.ravel()[second]
    cut_mismatch = np.maximum(mismatch.ravel()[first[crosses]], mismatch.ravel()[second[crosses]])
    cut_gradient = np.maximum(gradient.ravel()[first[crosses]], gradient.ravel()[second[crosses]])
    report = {
        "backend": backend, "cutEnergy": flow_value, "feasibleCutEnergy": feasible_energy,
        "hardPixels": int(hard.sum()), "pureSourcePixels": int((weight == 1).sum()),
        "mixedStaticPixels": int(mixed.sum()), "unknownWrittenPixels": 0,
        "cutRgbDifferenceP90": float(np.percentile(cut_mismatch, 90)),
        "cutGradientP90": float(np.percentile(cut_gradient, 90)),
        "noWarp": True, "noGeneratedBackground": True,
        "verifiedStaticMask": allowed_static,
    }
    return weight, mode, selected, report
