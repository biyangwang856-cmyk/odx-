"""Bounded planar road candidates from independent multi-view SfM tracks.

This is a geometry experiment, not automatic road classification. Input XYZ
must be triangulated from photographs; a DSM, depth map or plane sampled into
points is not independent evidence. A fixed spatial split is assigned before
fitting. Every qualified holdout is scored, including gross height outliers.
Each call represents one separately declared road segment/carriageway.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math

import numpy as np


@dataclass(frozen=True)
class RoadSurfaceConfig:
    split_cell_size_m: float = 5.0
    split_origin_xy: tuple[float, float] = (0.0, 0.0)
    min_views: int = 2
    min_fit_points: int = 12
    min_holdout_points: int = 6
    min_fit_cells: int = 4
    min_holdout_cells: int = 3
    min_axis_span_m: float = 2.0
    min_point_spacing_m: float = 0.10
    ransac_trials: int = 512
    random_seed: int = 17
    fit_inlier_threshold_m: float = 0.10
    min_fit_inlier_fraction: float = 0.65
    max_holdout_height_p90_m: float = 0.12
    max_holdout_height_m: float = 0.40
    max_holdout_reprojection_p90_px: float = 1.0
    max_holdout_reprojection_px: float = 3.0
    max_original_reprojection_p90_px: float = 2.0
    max_slope: float = 0.20
    max_support_distance_m: float = 8.0


def _validate_config(c):
    for name in ('min_views', 'min_fit_points', 'min_holdout_points', 'min_fit_cells',
                 'min_holdout_cells', 'ransac_trials'):
        value = getattr(c, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if c.min_views < 2 or c.min_fit_points < 3 or c.min_holdout_points < 3:
        raise ValueError('Independent multi-view fit and holdout are required')
    for name, value in asdict(c).items():
        if name in ('split_origin_xy', 'random_seed') or name.startswith('min_') and isinstance(value, int):
            continue
        if isinstance(value, (float, int)) and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if not 0 < c.min_fit_inlier_fraction <= 1:
        raise ValueError('Invalid fit consensus fraction')
    if np.asarray(c.split_origin_xy).shape != (2,) or not np.isfinite(c.split_origin_xy).all():
        raise ValueError('Invalid split origin')


def _hull(xy):
    # Monotone chain, no scipy dependency; retain only extreme CCW vertices.
    ordered = sorted(set(map(tuple, np.asarray(xy, np.float64))))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1], np.float64)


def _spread(xy):
    if len(xy) < 3:
        return 0.0
    centered = xy - np.mean(xy, axis=0)
    axes = np.linalg.svd(centered, full_matrices=False)[2]
    return float(np.min(np.ptp(centered @ axes.T, axis=0)))


@dataclass
class RoadSurfaceFit:
    report: dict
    coefficients: np.ndarray | None
    origin_xy: np.ndarray | None
    support_xy: np.ndarray
    hull_xy: np.ndarray
    max_support_distance_m: float

    def evaluate(self, xy, *, allow_candidate=False):
        """Return supported heights only; NaN outside the training support.

        An explicit diagnostic request permits a semantically unreviewed but
        geometrically accepted candidate. Failed geometry is never evaluable.
        This method does not authorize changing an orthophoto or existing DSM.
        """
        if not self.report['geometryAccepted'] or (not self.report['accepted'] and not allow_candidate):
            raise ValueError('Road surface is not accepted; candidate use must be explicit')
        xy = np.asarray(xy, np.float64)
        if xy.ndim < 1 or xy.shape[-1] != 2:
            raise ValueError('Expected XY coordinates with final dimension 2')
        flat = xy.reshape(-1, 2)
        heights = np.full(len(flat), np.nan, np.float64)
        for start in range(0, len(flat), 8192):
            p = flat[start:start+8192]
            keep = np.isfinite(p).all(axis=1)
            for a, b in zip(self.hull_xy, np.roll(self.hull_xy, -1, axis=0)):
                keep &= (b[0]-a[0])*(p[:, 1]-a[1])-(b[1]-a[1])*(p[:, 0]-a[0]) >= -1e-8
            nearest = np.full(len(p), np.inf)
            for s in self.support_xy:
                nearest = np.minimum(nearest, np.sum((p-s)**2, axis=1))
            keep &= nearest <= self.max_support_distance_m**2
            values = (p-self.origin_xy) @ self.coefficients[:2] + self.coefficients[2]
            heights[start:start+len(p)][keep] = values[keep]
        return heights.reshape(xy.shape[:-1])


def fit_road_surface(points, projectors, *, provenance, config=RoadSurfaceConfig()):
    """Fit one road plane and validate using untouched spatial holdout tracks.

    ``points`` contains JSON-compatible records with track_id, xyz (metres),
    origin='sfm-multiview-triangulation', and observations. Each observation
    contains image_id, pixel_xy, road_support ('reviewed' or 'candidate'),
    dynamic=False, and evidence_id binding its original-photo road assessment.
    Unknown/rejected/dynamic views do not support a road point. Projectors take
    Nx3 XYZ and return Nx2 calibrated image pixels in the declared frame.

    Provenance must identify the reconstruction SHA256 and image coordinate
    frame (raw-photo or undistorted-photo). Caller remains responsible for
    checking those input bytes, calibration, and semantic review evidence.
    SAM-only road_support='candidate' can never produce accepted=True.
    """
    _validate_config(config)
    digest = provenance.get('reconstruction_sha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest.lower()):
        raise ValueError('A reconstruction SHA256 is required')
    if provenance.get('image_coordinate_frame') not in ('raw-photo', 'undistorted-photo'):
        raise ValueError('Declare the actual photograph coordinate frame')
    points = list(points)
    seen, qualified, excluded, support_observations = set(), [], [], {}
    for p in sorted(points, key=lambda q: str(q.get('track_id', ''))):
        identity = str(p.get('track_id', ''))
        if not identity or identity in seen:
            raise ValueError('Track identities must be nonempty and unique')
        seen.add(identity)
        if p.get('origin') != 'sfm-multiview-triangulation':
            raise ValueError('DSM/depth/surface samples are not independent SfM road evidence')
        xyz = np.asarray(p.get('xyz'), np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Every XYZ must be a finite three-vector in metres')
        observations, images = [], set()
        for o in p.get('observations', []):
            image_id = o.get('image_id')
            if not image_id or image_id in images:
                raise ValueError('Each track must have at most one observation per photograph')
            images.add(image_id)
            pixel = np.asarray(o.get('pixel_xy'), np.float64)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError('Each raw observation needs a finite pixel coordinate')
            if o.get('road_support') not in ('reviewed', 'candidate') or o.get('dynamic') is not False:
                continue
            if not o.get('evidence_id') or image_id not in projectors:
                raise ValueError('Supported road observations need an evidence identity and calibrated projector')
            observations.append(o)
        if len(observations) < config.min_views:
            excluded.append({'trackId': identity, 'reason': 'insufficient-independent-road-views'})
            continue
        # Stable, XY-only thinning precedes any height or residual inspection.
        if any(np.linalg.norm(xyz[:2]-np.asarray(q['xyz'])[:2]) < config.min_point_spacing_m for q in qualified):
            excluded.append({'trackId': identity, 'reason': 'spatially-duplicate-track'})
            continue
        qualified.append(p)
        support_observations[identity] = observations
    config_dict = asdict(config)
    identity_record = {'points': points, 'provenance': provenance, 'config': config_dict}
    report = {'schema': 'ogs2.road-surface/v1', 'accepted': False, 'geometryAccepted': False,
              'scope': 'one declared road segment; no automatic road truth',
              'inputSha256': hashlib.sha256(json.dumps(identity_record, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'provenance': dict(provenance), 'config': config_dict, 'excluded': excluded,
              'fit': [], 'holdout': [], 'reasons': [], 'heightUnits': 'metres',
              'model': 'z=a*(x-origin_x)+b*(y-origin_y)+c',
              'holdoutUsedForModelSelection': False, 'holdoutOutliersRemoved': False,
              'semanticReviewAccepted': False}
    def failed(reason, coefficients=None, origin=None, support=None):
        report['reasons'].append(reason)
        empty = np.empty((0, 2), np.float64)
        return RoadSurfaceFit(report, coefficients, origin, empty if support is None else support,
                              empty if support is None else _hull(support), config.max_support_distance_m)
    if not qualified:
        return failed('no-qualified-independent-road-points')
    xyz = np.asarray([p['xyz'] for p in qualified], np.float64)
    cells = np.floor((xyz[:, :2]-config.split_origin_xy)/config.split_cell_size_m).astype(np.int64)
    heldout = (cells[:, 0] + 2*cells[:, 1]) % 4 == 0
    fit = ~heldout
    for i, p in enumerate(qualified):
        report['holdout' if heldout[i] else 'fit'].append({'trackId': str(p['track_id']), 'cell': cells[i].tolist()})
    if fit.sum() < config.min_fit_points or heldout.sum() < config.min_holdout_points:
        return failed('insufficient-preassigned-fit-or-holdout-points')
    if len(np.unique(cells[fit], axis=0)) < config.min_fit_cells or len(np.unique(cells[heldout], axis=0)) < config.min_holdout_cells:
        return failed('insufficient-independent-spatial-cells')
    if _spread(xyz[fit, :2]) < config.min_axis_span_m or _spread(xyz[heldout, :2]) < config.min_axis_span_m:
        return failed('insufficient-two-dimensional-road-support')
    origin = np.mean(xyz[fit, :2], axis=0)
    design = np.column_stack((xyz[:, :2]-origin, np.ones(len(xyz))))
    a, z = design[fit], xyz[fit, 2]
    rng = np.random.default_rng(config.random_seed)
    if math.comb(len(z), 3) <= config.ransac_trials:
        samples = list(itertools.combinations(range(len(z)), 3))
    else:
        samples = [rng.choice(len(z), 3, replace=False) for _ in range(config.ransac_trials)]
    best = None
    for sample in samples:
        sample = np.asarray(sample)
        if np.linalg.matrix_rank(a[sample]) < 3:
            continue
        coef = np.linalg.solve(a[sample], z[sample])
        residual = np.abs(a @ coef-z)
        inliers = residual <= config.fit_inlier_threshold_m
        score = (int(inliers.sum()), -float(np.median(residual[inliers])))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0][0] < config.min_fit_points or best[0][0]/len(z) < config.min_fit_inlier_fraction:
        return failed('insufficient-training-plane-consensus')
    inliers = best[1]
    coef = np.linalg.lstsq(a[inliers], z[inliers], rcond=None)[0]
    # Refit only the training consensus, never rescue from holdout results.
    support = xyz[fit, :2][inliers]
    report.update(coefficients=coef.tolist(), originXY=origin.tolist(),
                  fitInlierFraction=float(inliers.mean()), fitInlierPoints=int(inliers.sum()),
                  slope=float(np.linalg.norm(coef[:2])))
    for rec, is_inlier, residual in zip(report['fit'], inliers, np.abs(a @ coef-z)):
        rec.update(inlier=bool(is_inlier), heightResidualM=float(residual))
    if _spread(support) < config.min_axis_span_m or len(_hull(support)) < 3:
        return failed('insufficient-two-dimensional-training-consensus', coef, origin, support)
    height_errors = np.abs(design[heldout] @ coef-xyz[heldout, 2])
    corrected_errors, original_errors = [], []
    for rec, index, error in zip(report['holdout'], np.flatnonzero(heldout), height_errors):
        p = qualified[index]
        corrected = xyz[index].copy()
        corrected[2] = design[index] @ coef
        measurements = []
        for o in support_observations[str(p['track_id'])]:
            projected = np.asarray(projectors[o['image_id']](np.stack((xyz[index], corrected))), np.float64)
            if projected.shape != (2, 2) or not np.isfinite(projected).all():
                raise ValueError('Calibrated projection returned invalid photograph coordinates')
            errors = np.linalg.norm(projected-np.asarray(o['pixel_xy']), axis=1)
            original_errors.append(float(errors[0]))
            corrected_errors.append(float(errors[1]))
            measurements.append({'imageId': o['image_id'], 'originalErrorPx': float(errors[0]),
                                 'planeErrorPx': float(errors[1]), 'roadSupport': o['road_support'],
                                 'evidenceId': o['evidence_id']})
        rec.update(heightResidualM=float(error), observations=measurements)
    report.update(holdoutHeightP90M=float(np.percentile(height_errors, 90)),
                  holdoutHeightMaxM=float(np.max(height_errors)),
                  holdoutReprojectionP90Px=float(np.percentile(corrected_errors, 90)),
                  holdoutReprojectionMaxPx=float(np.max(corrected_errors)),
                  originalHoldoutReprojectionP90Px=float(np.percentile(original_errors, 90)))
    if report['slope'] > config.max_slope:
        report['reasons'].append('road-slope-exceeds-declared-prior')
    if report['holdoutHeightP90M'] > config.max_holdout_height_p90_m or report['holdoutHeightMaxM'] > config.max_holdout_height_m:
        report['reasons'].append('independent-height-holdout-failed')
    if report['holdoutReprojectionP90Px'] > config.max_holdout_reprojection_p90_px or report['holdoutReprojectionMaxPx'] > config.max_holdout_reprojection_px:
        report['reasons'].append('independent-photo-reprojection-holdout-failed')
    if report['originalHoldoutReprojectionP90Px'] > config.max_original_reprojection_p90_px:
        report['reasons'].append('original-sfm-photo-reprojection-failed')
    report['geometryAccepted'] = not report['reasons']
    # Every used track must have independent reviewed views. Candidate model
    # scores do not become road truth merely because the plane is flat.
    semantic = all(sum(o['road_support'] == 'reviewed' for o in support_observations[str(p['track_id'])]) >= config.min_views for p in qualified)
    report['semanticReviewAccepted'] = semantic
    report['accepted'] = report['geometryAccepted'] and semantic
    if not semantic:
        report['reasons'].append('road-semantics-remain-candidate')
    return RoadSurfaceFit(report, coef, origin, support, _hull(support), config.max_support_distance_m)
