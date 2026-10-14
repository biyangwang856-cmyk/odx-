"""Ordered photograph join with verified foreground-free cut edges.

A row can move horizontally through a clear two-row corridor. This avoids
the arbitrary one-pixel-per-row slope restriction of a usual seam DP.
"""

import numpy as np


def sample_cached_body(guard, uv):
    """Separate observed detections from missing cached mask context.

    Missing context remains forbidden for donor selection, but must never
    be connected into a fictitious vehicle by foreground component logic.
    """
    from model_engine.ogs2_corridor_projection import bilinear_support

    guard = np.asarray(guard, bool)
    known = bilinear_support(np.ones_like(guard), uv)
    body = (~bilinear_support(~guard, uv)) & known
    return body, ~known


def preserve_boundary_foreground(replacement, body, valid, road, domain, *, padding_pixels=3):
    """Close an exterior boundary around complete foreground cohorts.

    Include the whole cohort only when a road object is entirely observed
    inside the permitted domain. Otherwise keep its whole region original.
    This does not synthesise missing observations or certify detections.
    """
    import cv2

    replacement = np.asarray(replacement, bool).copy()
    arrays = [np.asarray(x, bool) for x in (body, valid, road, domain)]
    body, valid, road, domain = arrays
    if any(x.shape != replacement.shape for x in arrays):
        raise ValueError("Foreground closure masks must share one grid")
    if padding_pixels < 1:
        raise ValueError("A positive clearance collar is required")
    k = 2 * padding_pixels + 1
    envelope = cv2.dilate(body.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(envelope.astype(np.uint8), 8)
    touched = np.unique(labels[cut_edges(replacement) & body])
    actions = []
    for component in touched:
        if component == 0:
            continue
        x, y, w, h, area = stats[component]
        ys = slice(y, y + h)
        xs = slice(x, x + w)
        cohort = labels[ys, xs] == component
        on_road = bool(np.any(cohort & body[ys, xs] & road[ys, xs]))
        observed = bool(np.all(valid[ys, xs][cohort]))
        permitted = bool(np.all(domain[ys, xs][cohort]))
        include = on_road and observed and permitted
        replacement[ys, xs][cohort] = include
        actions.append(
            dict(
                component=int(component),
                pixels=int(area),
                wholeRoadForegroundIncluded=include,
                fullyObserved=observed,
                withinPermittedDomain=permitted,
            )
        )
    cut_count = int((cut_edges(replacement) & body).sum())
    if cut_count:
        raise AssertionError("Exterior closure splits a known foreground cohort")
    return replacement, dict(
        mode="whole foreground enclosure; incomplete or out-of-domain cohorts kept original",
        cohorts=actions,
        detectedBodyCutEndpoints=cut_count,
    )


def cut_edges(labels):
    """Both endpoints of every four-neighbour label discontinuity."""
    labels = np.asarray(labels, dtype=bool)
    edges = np.zeros_like(labels)
    different = labels[:, 1:] != labels[:, :-1]
    edges[:, 1:] |= different
    edges[:, :-1] |= different
    different = labels[1:] != labels[:-1]
    edges[1:] |= different
    edges[:-1] |= different
    return edges


def _relax_row(entry_cost, entry_index, edge_cost, allowed):
    """Multi-source shortest paths on a chain; blocked edges split the chain."""
    cost = entry_cost.copy()
    origin = entry_index.copy()
    if not len(allowed):
        return cost, origin
    starts = np.flatnonzero(allowed & np.r_[True, ~allowed[:-1]])
    ends = np.flatnonzero(allowed & np.r_[~allowed[1:], True]) + 2
    for lo, hi in zip(starts, ends):
        distances = np.r_[0.0, np.cumsum(edge_cost[lo : hi - 1], dtype=np.float64)]
        values = entry_cost[lo:hi] - distances
        prefix = np.minimum.accumulate(values)
        indices = np.maximum.accumulate(np.where(values <= prefix, np.arange(hi - lo), 0))
        candidate = prefix + distances
        improve = candidate < cost[lo:hi]
        cost[lo:hi][improve] = candidate[improve]
        origin[lo:hi][improve] = entry_index[lo:hi][indices[improve]]
        values = entry_cost[lo:hi][::-1] + distances[::-1]
        prefix = np.minimum.accumulate(values)
        indices = np.maximum.accumulate(np.where(values <= prefix, np.arange(hi - lo), 0))
        candidate = (prefix - distances[::-1])[::-1]
        improve = candidate < cost[lo:hi]
        cost[lo:hi][improve] = candidate[improve]
        origin[lo:hi][improve] = entry_index[lo:hi][::-1][indices][::-1][improve]
    return cost, origin


def select_highway_join(left, right, valid, protected, *, horizontal_penalty=0.2, penalty=None):
    """Return a hard left/right partition, or None when no safe path exists.

    True takes the right photograph. Every actual cut endpoint must be valid
    and outside ``protected``. No blending or fallback cut is performed.
    """
    left, right = np.asarray(left), np.asarray(right)
    valid, protected = np.asarray(valid, bool), np.asarray(protected, bool)
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("Expected equal H x W x 3 RGB arrays")
    if valid.shape != left.shape[:2] or protected.shape != valid.shape:
        raise ValueError("Join masks must match the RGB grid")
    h, w = valid.shape
    if h == 0 or w < 2:
        return None
    clear = valid & ~protected
    # A boundary j lies between pixels j and j+1, so both must be safe.
    node_allowed = clear[:, :-1] & clear[:, 1:]
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean(-1)
    if penalty is not None:
        penalty = np.asarray(penalty, np.float32)
        if penalty.shape != valid.shape or np.any(~np.isfinite(penalty)) or np.any(penalty < 0):
            raise ValueError("Join penalties must be finite nonnegative values on the RGB grid")
        difference += penalty
    cost = 0.5 * (difference[:, :-1] + difference[:, 1:]) + 1.0
    cost[~node_allowed] = np.inf
    best = cost[0].astype(np.float64)
    back = np.zeros((h, w - 1), np.int32)
    index = np.arange(w - 1, dtype=np.int32)
    for y in range(1, h):
        # Moving a boundary from j to j+1 changes the vertical cut at pixel
        # j+1. Both rows at that pixel must therefore be unprotected/valid.
        allowed = clear[y - 1, 1:-1] & clear[y, 1:-1]
        edge_cost = 0.5 * (difference[y - 1, 1:-1] + difference[y, 1:-1]) + horizontal_penalty
        relaxed, origins = _relax_row(best, index, edge_cost, allowed)
        best = relaxed + cost[y]
        back[y] = origins
    if not np.isfinite(best.min()):
        return None
    line = np.empty(h, np.int32)
    line[-1] = best.argmin()
    for y in range(h - 1, 0, -1):
        line[y - 1] = back[y, line[y]]
    labels = np.arange(w)[None, :] > line[:, None]
    if np.any(cut_edges(labels) & ~clear):
        raise AssertionError("Join path created an unsafe cut endpoint")
    return labels
