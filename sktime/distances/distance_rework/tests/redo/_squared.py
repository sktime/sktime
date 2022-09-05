# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework.tests.redo import BaseDistance, DistanceCallable


class _SquaredDistance(BaseDistance):
    _has_cost_matrix = False
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _independent_distance(
        self, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> DistanceCallable:
        def _numba_squared(_x, _y):
            x_size = _x.shape[0]
            distance = 0
            for i in range(x_size):
                distance += (_x[i] - _y[i]) ** 2
            return distance

        return _numba_squared
