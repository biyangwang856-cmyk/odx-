"""Small RGB offsets fitted outside objects, with fixed spatial validation.

This adjusts observed colours; it does not infer background or repair geometry.
The caller supplies measured static support and keeps its semantic uncertainty.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass(frozen=True)
class RoadColorFit:
    accepted: bool
    offset: np.ndarray | None
    report: dict

    def apply(self, image):
        if not self.accepted or self.offset is None:
            raise ValueError('Rejected colour correction')
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Require uint8 RGB')
        return np.clip(np.rint(image.astype(float) + self.offset), 0, 255).astype(np.uint8)

def fit_road_color_offset(anchor, source, static, *, max_offset=12.0,
                          cell_size=16, sample_step=4, max_holdout_p90=25.0):
    a, b, support = np.asarray(anchor), np.asarray(source), np.asarray(static)
    if (a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3 or b.dtype != np.uint8
            or b.shape != a.shape or support.dtype != bool or support.shape != a.shape[:2]):
        raise ValueError('Require matched uint8 RGB and boolean static support')
    if (not np.isfinite(max_offset) or not 0 < max_offset <= 12
            or not np.isfinite(max_holdout_p90) or max_holdout_p90 <= 0
            or isinstance(cell_size, bool) or not isinstance(cell_size, int) or cell_size < 4
            or isinstance(sample_step, bool) or not isinstance(sample_step, int) or sample_step < 1):
        raise ValueError('Invalid colour calibration limits')
    y, x = np.nonzero(support)
    use = (x % sample_step == 0) & (y % sample_step == 0)
    x, y = x[use], y[use]
    cells = np.column_stack([x // cell_size, y // cell_size])
    held = (cells[:, 0] + 2*cells[:, 1]) % 4 == 0
    train = ~held
    report = dict(schema='ogs2-road-colour-offset/v1', geometryCertified=False,
                  semanticsCertified=False, rgbExposureAdjusted=True, rgbBlended=False,
                  correctionKind='bounded-channel-offset', maxOffset=max_offset,
                  allHoldoutEvaluated=True, holdoutUsedForFitting=False,
                  trainPixels=int(train.sum()), holdoutPixels=int(held.sum()),
                  staticMaskSha256=hashlib.sha256(np.ascontiguousarray(support).tobytes()).hexdigest())
    def reject(reason, offset=None):
        return RoadColorFit(False, offset, {**report, 'accepted':False, 'reason':reason})
    if train.sum() < 48 or held.sum() < 24:
        return reject('insufficient-independent-colour-support')
    if len(np.unique(cells[train], axis=0)) < 3 or len(np.unique(cells[held], axis=0)) < 2:
        return reject('spatially-concentrated-colour-support')
    left, right = a[y, x].astype(float), b[y, x].astype(float)
    offset = np.median(left[train]-right[train], axis=0)
    report['offsetRGB'] = offset.tolist()
    if np.max(np.abs(offset)) > max_offset:
        return reject('colour-correction-exceeds-bound', offset)
    corrected = np.clip(np.rint(right[held]+offset), 0, 255)
    before = np.mean(np.abs(left[held]-right[held]), axis=1)
    after = np.mean(np.abs(left[held]-corrected), axis=1)
    report.update(beforeMedian=float(np.median(before)), afterMedian=float(np.median(after)),
                  beforeP90=float(np.percentile(before,90)), afterP90=float(np.percentile(after,90)),
                  holdoutDifferences=after.tolist())
    if report['afterP90'] > max_holdout_p90 or report['afterP90'] > report['beforeP90'] + .5:
        return reject('independent-colour-holdout-failed', offset)
    if report['afterMedian'] > report['beforeMedian'] + .1:
        return reject('correction-worsens-holdout-median', offset)
    return RoadColorFit(True, offset, {**report, 'accepted':True, 'reason':'bounded-colour-offset-validated'})
