# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np

from sktime.distances.distance_rework import BaseDistance, DistanceCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix


class _LcssDistance(BaseDistance):
    _has_cost_matrix = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _independent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1.0,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict,
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework import _SquaredDistance

        local_squared_distance = _SquaredDistance().distance_factory(
            x[0], y[0], strategy="local"
        )

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        def _numba_lcss(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[0]
            y_size = _y.shape[0]
            cost_matrix = np.zeros((x_size + 1, y_size + 1))

            for i in range(1, x_size + 1):
                for j in range(1, y_size + 1):
                    if np.isfinite(_bounding_matrix[i - 1, j - 1]):
                        squared_dist = local_squared_distance(_x[i - 1], _y[j - 1])

                        if squared_dist <= epsilon:
                            cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                        else:
                            cost_matrix[i, j] = max(
                                cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                            )
            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_lcss

    def _dependent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1.0,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict,
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework import _EuclideanDistance

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        _example_x = x[:, 0]
        _example_y = y[:, 0]
        euclidean_distance = _EuclideanDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        def _numba_lcss(_x, _y):
            x_size = _x.shape[1]
            y_size = _y.shape[1]
            cost_matrix = np.zeros((x_size + 1, y_size + 1))

            for i in range(1, x_size + 1):
                for j in range(1, y_size + 1):
                    if np.isfinite(_bounding_matrix[i - 1, j - 1]):
                        euclidean_dist = euclidean_distance(_x[:, i - 1], _y[:, j - 1])

                        if euclidean_dist <= epsilon:
                            cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                        else:
                            cost_matrix[i, j] = max(
                                cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                            )

            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_lcss

    def _result_distance_callback(self) -> Callable[[float], float]:
        def _result_callback(distance: float, x_size: int, y_size: int) -> float:
            return 1 - float(distance / min(x_size, y_size))

        return _result_callback
