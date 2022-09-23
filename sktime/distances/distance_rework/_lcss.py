# -*- coding: utf-8 -*-
from typing import Callable, Tuple, List

import numpy as np

from sktime.distances.distance_rework import (
    ElasticDistance,
    DistanceCallable,
)
from sktime.distances.distance_rework._base import AlignmentPathCallable


class _LcssDistance(ElasticDistance):
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
        from sktime.distances.distance_rework import _EuclideanDistance

        local_squared_distance = _EuclideanDistance().distance_factory(
            x[0], y[0], strategy="local"
        )

        _bounding_matrix = self._get_bounding_matrix(
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

        _bounding_matrix = self._get_bounding_matrix(
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

            return cost_matrix[-1, -1], cost_matrix

        return _numba_lcss

    def _result_distance_callback(self) -> Callable[[float], float]:
        def _result_callback(distance: float, x_size: int, y_size: int) -> float:
            return 1 - float(distance / min(x_size, y_size))

        return _result_callback

    def _alignment_path_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            strategy: str = 'independent',
            epsilon: float = 1.0,
            **kwargs
    ) -> AlignmentPathCallable:
        from sktime.distances.distance_rework import _EuclideanDistance

        _example_x = x[-1]
        _example_y = y[-1]
        euclidean_distance = _EuclideanDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        def _compute_min_return_path(
                _x: np.ndarray,
                _y: np.ndarray,
                _cost_matrix: np.ndarray,
                _bounding_matrix: np.ndarray
        ) -> List[Tuple]:
            x_size = _x.shape[1]
            y_size = _y.shape[1]

            i, j = (x_size, y_size)
            path = []

            while i > 0 and j > 0:
                if np.isfinite(_bounding_matrix[i - 1, j - 1]):
                    if euclidean_distance(_x[:, i - 1], _y[:, j - 1]) <= epsilon:
                        path.append((i - 1, j - 1))
                        i, j = (i - 1, j - 1)
                    elif _cost_matrix[i - 1][j] > _cost_matrix[i][j - 1]:
                        i = i - 1
                    else:
                        j = j - 1
            return path[::-1]

        return _compute_min_return_path
