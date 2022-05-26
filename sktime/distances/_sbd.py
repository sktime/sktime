# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Any, Union

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _SBDistance(NumbaDistance):

    def _distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: float = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            **kwargs: Any
    ) -> DistanceCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        def numba_dtw_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> float:
            cost_matrix = _shape_extraction(_x, _y, _bounding_matrix)
            return cost_matrix[-1, -1]

        return numba_dtw_distance


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

from scipy.signal import correlate2d

def _shape_extraction(
        x: np.ndarray,
        y: np.ndarray,
        bounding_matrix: np.ndarray,
) -> Union[float, np.ndarray]:
    x_size = x.shape[1]

    # Perform cross correlation
    test_x = x.transpose()
    test_y = y.transpose()
    cc = correlate2d(x, y)
    cctest = correlate2d(test_x, test_y)

    # Normalise cross correlation
    denom = np.array(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    ncc = cc / denom[:, np.newaxis]

    idx = ncc.argmax()
    shift = idx - x_size
    dist = 1 - ncc[:, idx]
    yshift = roll_zeropad(y, (idx + 1) - x_size)

    return dist, yshift

