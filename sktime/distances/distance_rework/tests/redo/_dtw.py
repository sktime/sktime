# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework.tests.redo import (
    BaseDistance,
    DistanceCallable,
)
from sktime.distances.lower_bounding import resolve_bounding_matrix


class _DtwDistance(BaseDistance):
    _has_cost_matrix = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _independent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict
    ) -> DistanceCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        def _numba_dtw(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[0]
            y_size = _y.shape[0]
            cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
            cost_matrix[0, 0] = 0.0

            for i in range(x_size):
                for j in range(y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        squared_dist = (_x[i] - _y[j]) ** 2
                        cost_matrix[i + 1, j + 1] = squared_dist + min(
                            cost_matrix[i, j + 1],
                            cost_matrix[i + 1, j],
                            cost_matrix[i, j],
                        )

            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_dtw

    def _dependent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework.tests.redo import _SquaredDistance

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        _example_x = x[:, 0]
        _example_y = y[:, 0]
        squared_distance = _SquaredDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        def _numba_dtw(_x, _y):
            x_size = _x.shape[1]
            y_size = _y.shape[1]
            cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
            cost_matrix[0, 0] = 0.0

            for i in range(x_size):
                for j in range(y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        squared_dist = squared_distance(_x[:, i], _y[:, j])
                        cost_matrix[i + 1, j + 1] = squared_dist + min(
                            cost_matrix[i, j + 1],
                            cost_matrix[i + 1, j],
                            cost_matrix[i, j],
                        )

            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_dtw
