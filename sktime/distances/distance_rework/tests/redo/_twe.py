# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np

from sktime.distances.distance_rework.tests.redo import BaseDistance, DistanceCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix


class _TweDistance(BaseDistance):
    _has_cost_matrix = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _preprocessing_time_series_callback(
        self, **kwargs
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _preprocessing_time_series(_x: np.ndarray) -> np.ndarray:
            padded_x = np.zeros((_x.shape[0], _x.shape[1] + 1))
            zero_arr = np.array([0.0])
            for i in range(_x.shape[0]):
                padded_x[i, :] = np.concatenate((zero_arr, _x[i, :]))
            return padded_x

        return _preprocessing_time_series

    def _independent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
        **kwargs: dict
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework.tests.redo import _SquaredDistance

        local_squared_dist = _SquaredDistance().distance_factory(
            x[0], y[0], strategy="local"
        )

        pad_ts = self._preprocessing_time_series_callback(**kwargs)

        _bounding_matrix = resolve_bounding_matrix(
            pad_ts(x), pad_ts(y), window, itakura_max_slope, bounding_matrix
        )

        def _numba_twe(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[0]
            y_size = _y.shape[0]
            cost_matrix = np.zeros((x_size, y_size))
            cost_matrix[0, 1:] = np.inf
            cost_matrix[1:, 0] = np.inf

            del_add = nu + lmbda

            for i in range(1, x_size):
                for j in range(1, y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        # Deletion in x
                        del_x_squared_dist = local_squared_dist(_x[i], _x[i - 1])
                        del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add

                        # Deletion in y
                        del_y_squared_dist = local_squared_dist(_y[j], _y[j - 1])
                        del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                        # Match
                        match_same_squared_d = local_squared_dist(_x[i], _y[j])
                        match_prev_squared_d = local_squared_dist(_x[i - 1], _y[j - 1])
                        match = (
                            cost_matrix[i - 1, j - 1]
                            + match_same_squared_d
                            + match_prev_squared_d
                            + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                        )
                        cost_matrix[i, j] = min(del_x, del_y, match)

            return cost_matrix[-1, -1], cost_matrix

        return _numba_twe

    def _dependent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
        **kwargs: dict
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework.tests.redo import _EuclideanDistance

        pad_ts = self._preprocessing_time_series_callback(**kwargs)

        _bounding_matrix = resolve_bounding_matrix(
            pad_ts(x), pad_ts(y), window, itakura_max_slope, bounding_matrix
        )

        _example_x = x[:, 0]
        _example_y = y[:, 0]
        euclidean_distance = _EuclideanDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        def _numba_twe(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[1]
            y_size = _y.shape[1]
            cost_matrix = np.zeros((x_size, y_size))
            cost_matrix[0, 1:] = np.inf
            cost_matrix[1:, 0] = np.inf

            del_add = nu + lmbda

            for i in range(1, x_size):
                for j in range(1, y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        # Deletion in x
                        del_x_squared_dist = euclidean_distance(_x[:, i - 1], _x[:, i])
                        del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                        # Deletion in y
                        del_y_squared_dist = euclidean_distance(_y[:, j - 1], _y[:, j])
                        del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                        # Match
                        match_same_squared_d = euclidean_distance(_x[:, i], _y[:, j])
                        match_prev_squared_d = euclidean_distance(
                            _x[:, i - 1], _y[:, j - 1]
                        )
                        match = (
                            cost_matrix[i - 1, j - 1]
                            + match_same_squared_d
                            + match_prev_squared_d
                            + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                        )

                        cost_matrix[i, j] = min(del_x, del_y, match)

            return cost_matrix[-1, -1], cost_matrix

        return _numba_twe
