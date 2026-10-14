"""Auditable whole-road background replacement from registered photographs.

No generative fill or alpha averaging is used. Each changed RGB pixel is
copied from one measured photograph, with a separately recorded bounded
constant colour offset. Masks and registration remain external evidence:
this selector cannot certify that a detector missed no foreground object.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ogs2_road_patch import select_road_patch


@dataclass(frozen=True)
class HighwayBackgroundConfig:
    colour_offset_limit: int = 12
    temporal_difference: float = 30.0
    stable_background_mad: float = 12.0
    min_other_observations: int = 3
    shadow_padding_pixels: int = 25
    patch_collar_pixels: int = 40
    maximum_component_pixels: int = 400000


def select_background(images, valid, guards, road, *, config=HighwayBackgroundConfig()):
    images = np.asarray(images)
    valid = np.asarray(valid)
    guards = np.asarray(guards)
    road = np.asarray(road)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Require N x H x W x 3 uint8 photographs on one grid")
    n, h, w, _ = images.shape
    if n < 1 or valid.shape != (n, h, w) or guards.shape != valid.shape or road.shape != (h, w):
        raise ValueError("Mask shapes must match the common grid")
    if any(v.dtype != np.bool_ for v in [valid, guards, road]):
        raise ValueError("Masks must be boolean")
    if min(h, w) < 3 or not road.any():
        raise ValueError("Nonempty road and image collar required")
    if config.colour_offset_limit < 0 or config.colour_offset_limit > 12:
        raise ValueError("Colour correction must stay within 12 byte levels")
    if config.min_other_observations < 3:
        raise ValueError("Temporal support needs at least three observations")
    adjusted = images.copy()
    offsets = []
    for i in range(n):
        common = road & valid[0] & valid[i] & ~guards[0] & ~guards[i]
        offset = np.zeros(3, np.int16)
        if common.sum() >= 100:
            delta = images[0][common].astype(np.int16) - images[i][common].astype(np.int16)
            offset = np.clip(
                np.rint(np.median(delta, axis=0)), -config.colour_offset_limit, config.colour_offset_limit
            ).astype(np.int16)
        adjusted[i] = np.clip(images[i].astype(np.int16) + offset, 0, 255).astype(np.uint8)
        offsets.append(offset.tolist())
    clear = valid & ~guards
    # The temporal reference guides foreground proposals only. It is never
    # written into the orthophoto and is not proof that a road is empty.
    stack = np.where(clear[..., None], adjusted.astype(np.float32), np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - reference[None]), axis=0).mean(-1)
    count = clear.sum(0)
    stable = (
        np.isfinite(reference).all(-1)
        & (mad <= config.stable_background_mad)
        & (count >= config.min_other_observations + 1)
        & road
    )
    bad = guards.copy()
    temporal = []
    for i in range(n):
        diff = np.abs(adjusted[i].astype(np.float32) - reference).mean(-1)
        anomaly = (diff > config.temporal_difference) & stable & valid[i]
        anomaly = cv2.morphologyEx(anomaly.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cc, lab, stats, _ = cv2.connectedComponentsWithStats(anomaly, 8)
        keep = (stats[:, cv2.CC_STAT_AREA] >= 15) & (
            stats[:, cv2.CC_STAT_AREA] <= config.maximum_component_pixels
        )
        keep[0] = False
        anomaly = keep[lab] & road
        bad[i] |= anomaly
        temporal.append(int(anomaly.sum()))
    if config.shadow_padding_pixels:
        k = config.shadow_padding_pixels * 2 + 1
        bad = np.stack(
            [
                cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
                for m in bad
            ]
        )
    output = adjusted[0].copy()
    source = np.zeros((h, w), np.uint16)
    coverage = valid[0].copy()
    remaining = bad[0] & road
    changes = []
    failed = []
    # Source-coverage holes are addressed explicitly, never treated as black
    # road observations or mixed with valid pixels.
    for i in range(1, n):
        take = ~coverage & valid[i] & ~bad[i] & road
        output[take] = adjusted[i][take]
        source[take] = i
        coverage[take] = True
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(remaining.astype(np.uint8), 8)
    for component in sorted(range(1, ncc), key=lambda k: int(stats[k, cv2.CC_STAT_AREA]), reverse=True):
        component_left, t, cw, ch, area = map(int, stats[component])
        pad = config.patch_collar_pixels
        x0 = max(0, component_left - pad)
        x1 = min(w, component_left + cw + pad)
        y0 = max(0, t - pad)
        y1 = min(h, t + ch + pad)
        sl = np.s_[y0:y1, x0:x1]
        hard = (lab[sl] == component) & remaining[sl]
        if not hard.any():
            continue
        if area > config.maximum_component_pixels or hard[[0, -1], :].any() or hard[:, [0, -1]].any():
            failed.append(
                dict(component=component, reason="component-too-large-or-touches-image-boundary", pixels=area)
            )
            continue
        candidates = []
        for i in range(1, n):
            allowed = valid[i][sl] & ~bad[i][sl] & road[sl]
            if (hard & ~allowed).any():
                continue
            collar = cv2.dilate(hard.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
            collar &= ~hard & allowed & coverage[sl] & ~remaining[sl]
            difference = np.abs(output[sl].astype(np.float32) - adjusted[i][sl].astype(np.float32)).mean(-1)
            score = float(np.percentile(difference[collar], 90)) if collar.sum() >= 20 else float("inf")
            candidates.append((score, i, allowed))
        applied = False
        for score, i, allowed in sorted(candidates, key=lambda p: p[0]):
            if not np.isfinite(score) or score > 25:
                continue
            result = select_road_patch(
                output[sl],
                adjusted[i][sl],
                coverage[sl],
                valid[i][sl] & ~bad[i][sl],
                hard,
                allowed,
                np.zeros(hard.shape, bool),
            )
            if not result.accepted:
                continue
            take = result.take_candidate
            # Validate actual seam endpoints after optimisation. A feasible
            # discrete cut alone is not a photometric acceptance certificate.
            seam = result.seam_mask
            values = np.abs(output[sl].astype(np.float32) - adjusted[i][sl].astype(np.float32)).mean(-1)[seam]
            p90 = float(np.percentile(values, 90)) if len(values) else 0.0
            maximum = float(values.max()) if len(values) else 0.0
            if p90 > 25 or maximum > 80:
                continue
            output[sl][take] = adjusted[i][sl][take]
            source[sl][take] = i
            coverage[sl][take] = True
            remaining[sl][take] = False
            changes.append(
                dict(
                    component=component,
                    donor=i,
                    pixels=int(take.sum()),
                    seamP90=p90,
                    seamMax=maximum,
                    window=[x0, y0, x1 - x0, y1 - y0],
                )
            )
            applied = True
            break
        if not applied and n > 1:
            # Dense traffic can make an entire connected foreground proposal
            # impossible for any ONE donor. Several independently clear real
            # photographs may cover it jointly. Their masks are hard source
            # exclusions; no median RGB or inpainting enters the output.
            allowed = [coverage[sl] & ~hard]
            for i in range(1, n):
                m = valid[i][sl] & ~bad[i][sl] & road[sl]
                m = m.copy()
                m[[0, -1], :] = False
                m[:, [0, -1]] = False
                allowed.append(m)
            joint = np.any(np.stack(allowed[1:]), axis=0)
            if not (hard & ~joint).any():
                colour = [output[sl].copy()] + [adjusted[i][sl].copy() for i in range(1, n)]
                masks = [cv2.UMat((m * 255).astype(np.uint8)) for m in allowed]
                finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
                finder.find([v.astype(np.float32) for v in colour], [(0, 0)] * n, masks)
                partition = np.zeros(hard.shape, np.uint16)
                covered = np.zeros(hard.shape, bool)
                for i, m in enumerate(masks):
                    take = (m.get() > 0) & allowed[i] & ~covered
                    partition[take] = i
                    covered |= take
                # Mask borders can leave graph-cut holes. Only a valid hard
                # permitted source can fill them, with explicit provenance.
                for i, m in enumerate(allowed):
                    take = m & ~covered
                    partition[take] = i
                    covered |= take
                feasible = covered.all() and not ((partition == 0) & hard).any()
                values = []
                for axis in [0, 1]:
                    a = partition[:-1] if axis == 0 else partition[:, :-1]
                    b = partition[1:] if axis == 0 else partition[:, 1:]
                    y, x = np.nonzero(a != b)
                    for i, j in set(zip(a[y, x].tolist(), b[y, x].tolist())):
                        chosen = (a[y, x] == i) & (b[y, x] == j)
                        py, px = y[chosen], x[chosen]
                        values.extend(
                            np.abs(
                                colour[i][py, px].astype(np.float32) - colour[j][py, px].astype(np.float32)
                            )
                            .mean(-1)
                            .tolist()
                        )
                        qy, qx = py + (axis == 0), px + (axis == 1)
                        values.extend(
                            np.abs(
                                colour[i][qy, qx].astype(np.float32) - colour[j][qy, qx].astype(np.float32)
                            )
                            .mean(-1)
                            .tolist()
                        )
                p90 = float(np.percentile(values, 90)) if values else 0.0
                maximum = max(values, default=0.0)
                if feasible and p90 <= 25 and maximum <= 80:
                    take = partition > 0
                    for i in range(1, n):
                        use = partition == i
                        output[sl][use] = colour[i][use]
                        source[sl][use] = i
                    coverage[sl][take] = True
                    remaining[sl][take] = False
                    changes.append(
                        dict(
                            component=component,
                            donor="joint-real-photo-graph-cut",
                            pixels=int(take.sum()),
                            seamP90=p90,
                            seamMax=float(maximum),
                            window=[x0, y0, x1 - x0, y1 - y0],
                        )
                    )
                    applied = True
        if not applied:
            failed.append(
                dict(component=component, reason="no-clear-registered-donor-with-accepted-seam", pixels=area)
            )
    return dict(
        image=output,
        source=source,
        valid=coverage,
        bad=bad,
        remaining=remaining,
        report=dict(
            schema="ogs2.highway-background/v1",
            colourOffsets=offsets,
            temporalProposalPixels=temporal,
            applied=changes,
            unresolved=failed,
            unresolvedPixels=int(remaining.sum()),
            uncoveredRoadPixels=int((road & ~coverage).sum()),
            selectedDonorGuardPixels=int(
                np.take_along_axis(bad, source[None], axis=0)[0][road & (source > 0)].sum()
            ),
            inventedPixels=0,
            mixedPixels=0,
            semanticsCertified=False,
        ),
    )
