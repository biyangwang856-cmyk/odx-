"""Small, explicitly recorded source-sampling corrections on a fixed map grid.

Measured pairs are anchor/output coordinates and corresponding coordinates in
one projected source image. This module neither changes the geographic grid nor
certifies road semantics, absolute accuracy, empty background, or a whole mosaic.
Spatial holdout cells are assigned before fitting. The caller must not prefilter
measurements by their displacement or by a desired correction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class RoadRegistrationConfig:
    cell_size_px: float = 64.0
    cell_origin_xy: tuple[float, float] = (0.0, 0.0)
    min_train_points: int = 12
    min_holdout_points: int = 8
    min_inner_points: int = 6
    min_train_cells: int = 4
    min_holdout_cells: int = 3
    min_inner_cells: int = 2
    min_axis_span_px: float = 24.0
    max_displacement_px: float = 8.0
    min_jacobian_singular_value: float = 0.99
    max_jacobian_singular_value: float = 1.01
    max_holdout_p90_px: float = 1.0
    max_holdout_error_px: float = 2.0
    affine_min_improvement_px: float = 0.10
    affine_min_improvement_fraction: float = 0.10
    huber_delta_px: float = 1.0
    robust_iterations: int = 8


@dataclass(frozen=True)
class RoadRegistrationFit:
    accepted: bool
    matrix: np.ndarray | None
    report: dict

    def source_coordinates(self, output_xy, *, allow_diagnostic=False):
        """Return coordinates in the projected source, NOT raw camera pixels.

        Compose this mapping with that source's existing map-to-camera UV map.
        A rejected matrix can only be inspected with explicit diagnostic use.
        The caller must recompute source validity/masks after any resampling.
        """
        if self.matrix is None or (not self.accepted and not allow_diagnostic):
            raise ValueError('Only accepted corrections are usable without diagnostic override')
        xy = np.asarray(output_xy, np.float64)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Require finite XY coordinates')
        h, w = self.report['gridShape']
        if (xy < 0).any() or (xy > [w-1, h-1]).any():
            raise ValueError('Correction is only bounded on the declared output grid')
        return xy @ self.matrix[:, :2].T + self.matrix[:, 2]


def _config(c):
    for name, value in asdict(c).items():
        if name == 'cell_origin_xy':
            if np.asarray(value).shape != (2,) or not np.isfinite(value).all():
                raise ValueError('Invalid cell origin')
        elif name.startswith('min_') and name.endswith(('points', 'cells')) or name == 'robust_iterations':
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if c.min_train_points < 6 or c.min_inner_points < 3 or c.min_holdout_points < 3:
        raise ValueError('Insufficient independent fit/holdout minima')
    if not 0 < c.min_jacobian_singular_value <= 1 <= c.max_jacobian_singular_value:
        raise ValueError('Jacobian bounds must include the identity')
    if c.affine_min_improvement_fraction >= 1:
        raise ValueError('Invalid affine improvement fraction')


def _span(x):
    if len(x) < 3:
        return 0.0
    axes = np.linalg.svd(x-x.mean(axis=0), full_matrices=False)[2]
    return float(np.min(np.ptp(x @ axes.T, axis=0)))


def _fit(a, b, model, c):
    if model == 'translation':
        return np.column_stack((np.eye(2), np.median(b-a, axis=0)))
    origin = a.mean(axis=0)
    scale = max(float(np.max(np.ptp(a, axis=0))), 1.0)
    x = np.column_stack(((a-origin)/scale, np.ones(len(a))))
    if np.linalg.matrix_rank(x) < 3:
        return None
    weights = np.ones(len(a))
    # All fitting points remain present; only training residuals set Huber weights.
    for _ in range(c.robust_iterations):
        root = np.sqrt(weights)
        coefficient = np.linalg.lstsq(x*root[:, None], b*root[:, None], rcond=None)[0]
        residual = np.linalg.norm(x @ coefficient-b, axis=1)
        weights = np.minimum(1.0, c.huber_delta_px/np.maximum(residual, 1e-12))
    linear = coefficient[:2].T/scale
    return np.column_stack((linear, coefficient[2]-linear @ origin))


def _errors(m, a, b):
    return np.linalg.norm(a @ m[:, :2].T + m[:, 2]-b, axis=1)


def _bounds(m, shape, c):
    h, w = shape
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], np.float64)
    singular = np.linalg.svd(m[:, :2], compute_uv=False)
    maximum = float(np.max(_errors(m, corners, corners)))
    determinant = float(np.linalg.det(m[:, :2]))
    ok = (np.isfinite(m).all() and determinant > 0
          and maximum <= c.max_displacement_px
          and singular.min() >= c.min_jacobian_singular_value
          and singular.max() <= c.max_jacobian_singular_value)
    return bool(ok), {'maxDomainDisplacementPx': maximum,
                      'jacobianSingularValues': singular.tolist(), 'jacobianDeterminant': determinant}


def fit_road_registration(first, second, shape, *, config=RoadRegistrationConfig()):
    """Estimate output-grid -> projected-source coordinates from measured pairs.

    ``shape`` is (height, width) of the common, unresized orthophoto pixel grid.
    Caller-supplied train/holdout flags are intentionally unsupported. Outer
    holdout uses (cell_x + 2*cell_y) % 4 == 0. An independent inner split of the
    remaining cells selects translation versus affine. No outer holdout point
    is removed, reweighted, or used to select/refit a model.
    """
    _config(config)
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if (a.ndim != 2 or a.shape[1:] != (2,) or a.shape != b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Require finite paired Nx2 measurements')
    if (len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 3 for v in shape)):
        raise ValueError('Shape must contain integer height/width >= 3')
    h, w = map(int, shape)
    if len(a) and ((a < 0).any() or (b < 0).any() or (a > [w-1, h-1]).any() or (b > [w-1, h-1]).any()):
        raise ValueError('Measurements lie outside the actual input grid')
    if len(np.unique(np.round(a, 6), axis=0)) != len(a) or len(np.unique(np.round(b, 6), axis=0)) != len(b):
        raise ValueError('Duplicate source/target measurements cannot inflate evidence')
    cells = np.floor((a-config.cell_origin_xy)/config.cell_size_px).astype(np.int64)
    outer = (cells[:, 0]+2*cells[:, 1]) % 4 == 0
    train = ~outer
    # Different cell expression avoids merely repeating the outer partition.
    inner_validation = train & ((2*cells[:, 0]+cells[:, 1]) % 3 == 0)
    inner_fit = train & ~inner_validation
    payload = {'first': a.tolist(), 'second': b.tolist(), 'shape': [h, w], 'config': asdict(config)}
    report = {'schema': 'ogs2-road-registration/v1', 'accepted': False, 'reasons': [], 'gridShape': [h, w],
              'coordinateMapping': 'anchor-output-pixel -> projected-source-pixel',
              'geographicGridChanged': False, 'sourceSamplingCorrectionAppliedByCaller': True,
              'wholeOrthophotoCertified': False, 'semanticEvidenceProvided': False,
              'absolutePositionAccuracyCertified': False, 'rgbWarped': True,
              'outerHoldoutUsedForModelSelection': False, 'outerHoldoutOutliersRemoved': False,
              'config': asdict(config), 'inputSha256': hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'outerHoldoutIndices': np.flatnonzero(outer).tolist(),
              'trainingIndices': np.flatnonzero(train).tolist(),
              'innerFitIndices': np.flatnonzero(inner_fit).tolist(),
              'innerValidationIndices': np.flatnonzero(inner_validation).tolist(),
              'cells': cells.tolist(), 'modelCandidates': []}

    def finish(reason=None, matrix=None):
        if reason:
            report['reasons'].append(reason)
        report['accepted'] = not report['reasons']
        if matrix is not None:
            matrix = matrix.copy()
            matrix.setflags(write=False)
            report['sourceSamplingMatrix'] = matrix.tolist()
        return RoadRegistrationFit(report['accepted'], matrix, report)

    for label, mask, minimum, min_cells in (
            ('training', train, config.min_train_points, config.min_train_cells),
            ('outer-holdout', outer, config.min_holdout_points, config.min_holdout_cells),
            ('inner-fit', inner_fit, config.min_inner_points, config.min_inner_cells),
            ('inner-validation', inner_validation, config.min_inner_points, config.min_inner_cells)):
        report[label+'PointCount'] = int(mask.sum())
        report[label+'CellCount'] = len(np.unique(cells[mask], axis=0))
        if mask.sum() < minimum or report[label+'CellCount'] < min_cells:
            return finish('insufficient-independent-'+label)
        if _span(a[mask]) < config.min_axis_span_px:
            return finish('degenerate-or-concentrated-'+label)

    candidates = {}
    for model in ('translation', 'affine'):
        m = _fit(a[inner_fit], b[inner_fit], model, config)
        if m is None:
            report['modelCandidates'].append({'model': model, 'eligible': False, 'reason': 'rank-deficient'})
            continue
        eligible, bounds = _bounds(m, (h, w), config)
        errors = _errors(m, a[inner_validation], b[inner_validation])
        value = float(np.percentile(errors, 90))
        report['modelCandidates'].append({'model': model, 'eligible': eligible, 'innerValidationP90Px': value, **bounds})
        if eligible:
            candidates[model] = value
    if not candidates:
        return finish('no-bounded-inner-model')
    model = 'translation' if 'translation' in candidates else 'affine'
    if 'translation' in candidates and 'affine' in candidates:
        gain = candidates['translation']-candidates['affine']
        if gain >= max(config.affine_min_improvement_px, config.affine_min_improvement_fraction*candidates['translation']):
            model = 'affine'
    report['selectedModel'] = model
    m = _fit(a[train], b[train], model, config)
    if m is None:
        return finish('rank-deficient-final-training-fit')
    bounded, bounds = _bounds(m, (h, w), config)
    before = np.linalg.norm(b[outer]-a[outer], axis=1)
    after = _errors(m, a[outer], b[outer])
    report.update(bounds)
    report.update(beforeHoldoutP90Px=float(np.percentile(before, 90)), afterHoldoutP90Px=float(np.percentile(after, 90)),
                  beforeHoldoutMaxPx=float(before.max()), afterHoldoutMaxPx=float(after.max()),
                  holdoutResidualsPx=after.tolist(), allOuterHoldoutEvaluated=True)
    if not bounded:
        report['reasons'].append('final-correction-exceeds-domain-or-jacobian-bound')
    if report['afterHoldoutP90Px'] > config.max_holdout_p90_px or report['afterHoldoutMaxPx'] > config.max_holdout_error_px:
        report['reasons'].append('independent-outer-holdout-failed')
    if report['afterHoldoutP90Px'] > report['beforeHoldoutP90Px']+1e-9:
        report['reasons'].append('correction-worsens-independent-holdout')
    return finish(matrix=m)
