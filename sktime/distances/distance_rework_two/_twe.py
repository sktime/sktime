# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()


class _TweDistance(ElasticDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _preprocess_timeseries(x, *args):
        padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
        zero_arr = np.array([0.0])
        for i in range(x.shape[0]):
            padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
        return padded_x

    @staticmethod
    def _distance(
        x: np.ndarray,
        y: np.ndarray,
        window: Union[float, None] = None,
        itakura_max_slope: Union[float, None] = None,
        bounding_matrix: Union[np.ndarray, None] = None,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size, y_size))
        cost_matrix[0, 1:] = np.inf
        cost_matrix[1:, 0] = np.inf

        del_add = nu + lmbda

        for i in range(1, x_size):
            for j in range(1, y_size):
                if np.isfinite(bounding_matrix[i, j]):
                    # Deletion in x
                    del_x_squared_dist = squared_distance(x[:, i - 1], x[:, i])
                    del_x = cost_matrix[i - 1, j] + del_x_squared_dist + del_add
                    # Deletion in y
                    del_y_squared_dist = squared_distance(y[:, j - 1], y[:, j])
                    del_y = cost_matrix[i, j - 1] + del_y_squared_dist + del_add

                    # Match
                    match_same_squared_d = squared_distance(x[:, i], y[:, j])
                    match_prev_squared_d = squared_distance(x[:, i - 1], y[:, j - 1])
                    match = (
                        cost_matrix[i - 1, j - 1]
                        + match_same_squared_d
                        + match_prev_squared_d
                        + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                    )

                    cost_matrix[i, j] = min(del_x, del_y, match)

        return cost_matrix[-1, -1], cost_matrix

    @staticmethod
    def _alignment_path(
        x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray, bounding_matrix, *args
    ) -> List[Tuple]:
        x_size = x.shape[-1]
        y_size = y.shape[-1]

        for i in range(x_size):
            for j in range(y_size):
                if not np.isfinite(bounding_matrix[i, j]):
                    cost_matrix[i, j] = np.inf

        i = cost_matrix.shape[0] - 1
        j = cost_matrix.shape[1] - 1
        alignment = []
        while True:
            alignment.append((i - 1, j - 1))

            if alignment[-1] == (0, 0):
                break

            arr = np.array(
                [
                    cost_matrix[i - 1, j - 1],
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
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
