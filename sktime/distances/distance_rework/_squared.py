# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from sktime.distances.distance_rework import (
    BaseDistance,
    DistanceCallable,
    LocalDistanceCallable,
)


@njit(cache=True, fastmath=True)
def local_squared_distance(x: float, y: float):
    """Determine euclidean distance between two points.

    Parameters
    ----------
    x: float
        First value
    y: float
        Second value

    Returns
    -------
    float
        Euclidean distance between x and y

    """
    return ((x - y) ** 2) ** (1 / 2)


class _SquaredDistance(BaseDistance):
    _has_cost_matrix = False
    _has_local_distance = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _independent_distance(
        self, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> DistanceCallable:
        def _numba_squared(_x: np.ndarray, _y: np.ndarray):
            x_size = _x.shape[0]
            distance = 0
            for i in range(x_size):
                distance += (_x[i] - _y[i]) ** 2
            return distance

        return _numba_squared

    def _local_distance(self, x: float, y: float, **kwargs) -> LocalDistanceCallable:
        def _local_squared(_x: float, _y: float) -> float:
            return (_x - _y) ** 2

        return _local_squared
