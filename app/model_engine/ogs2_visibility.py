"""Bounded-memory surface ray checks for OGS2, without per-ray array allocation."""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def ray_visible(points, position, surface, inverse, distances, tolerance, step):
    height, width = points.shape[1:]
    sh, sw = surface.shape
    valid = np.zeros((height, width), np.bool_)
    for y in prange(height):
        for x in range(width):
            px, py, pz = points[0, y, x], points[1, y, x], points[2, y, x]
            if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):
                continue
            dx, dy, dz = position[0] - px, position[1] - py, position[2] - pz
            horizontal = max(math.sqrt(dx * dx + dy * dy), step)
            okay = True
            for distance in distances:
                fraction = min(distance / horizontal, 0.95)
                xx, yy, zz = px + dx * fraction, py + dy * fraction, pz + dz * fraction
                j = inverse[0] * xx + inverse[1] * yy + inverse[2] - 0.5
                i = inverse[3] * xx + inverse[4] * yy + inverse[5] - 0.5
                if j < 0 or i < 0 or j > sw - 1 or i > sh - 1:
                    okay = False
                    break
                col, row = int(math.floor(j)), int(math.floor(i))
                col2, row2 = min(col + 1, sw - 1), min(row + 1, sh - 1)
                fx, fy = j - col, i - row
                z00, z01 = surface[row, col], surface[row, col2]
                z10, z11 = surface[row2, col], surface[row2, col2]
                terrain = (z00 * (1 - fx) + z01 * fx) * (1 - fy) + (z10 * (1 - fx) + z11 * fx) * fy
                if not np.isfinite(terrain) or terrain > zz + tolerance:
                    okay = False
                    break
                if fraction >= 0.95:
                    break
            valid[y, x] = okay
    return valid
