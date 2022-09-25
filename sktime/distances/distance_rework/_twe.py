# -*- coding: utf-8 -*-
from typing import Callable, Tuple, List

import numpy as np

from sktime.distances.distance_rework import ElasticDistance, DistanceCallable
from sktime.distances.distance_rework._base import AlignmentPathCallable


class _TweDistance(ElasticDistance):
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
        from sktime.distances.distance_rework import _SquaredDistance

        local_squared_dist = _SquaredDistance().distance_factory(
            x[0], y[0], strategy="local"
        )

        _bounding_matrix = self._get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
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
        from sktime.distances.distance_rework import _EuclideanDistance

        _bounding_matrix = self._get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
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

    def _alignment_path_factory(
            self, x: np.ndarray, y: np.ndarray, strategy: str = 'independent', **kwargs
    ) -> AlignmentPathCallable:
        def _compute_min_return_path(
                _x: np.ndarray,
                _y: np.ndarray,
                _cost_matrix: np.ndarray,
                _bounding_matrix: np.ndarray
        ) -> List[Tuple]:
            x_size = _x.shape[-1]
            y_size = _y.shape[-1]

            for i in range(x_size):
                for j in range(y_size):
                    if not np.isfinite(_bounding_matrix[i, j]):
                        _cost_matrix[i, j] = np.inf

            i = _cost_matrix.shape[0] - 1
            j = _cost_matrix.shape[1] - 1
            alignment = []
            while True:
                alignment.append((i - 1, j - 1))

                if alignment[-1] == (0, 0):
                    break

                arr = np.array(
                    [
                        _cost_matrix[i - 1, j - 1],
                        _cost_matrix[i - 1, j],
                        _cost_matrix[i, j - 1],
                    ]
                )
                min_index = np.argmin(arr)

                if min_index == 0:
                    i = i - 1
                    j = j - 1
                elif min_index == 1:
                    i = i - 1
                else:
                    j = j - 1

            return alignment[::-1]

        return _compute_min_return_path
