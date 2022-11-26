# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np
from numba import njit

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()


@njit(cache=True, fastmath=True)
def _convert_2d(x: np.ndarray, *args):
    if x.ndim == 1:
        # Use this instead of numpy because weird numba errors sometimes with
        # np.reshape
        x_size = x.shape[0]
        _process_x = np.zeros((1, x_size))
        _process_x[0] = x
        return_val = _process_x
        return _process_x
    return x


class _LcssDistance(ElasticDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _distance(
        x: np.ndarray,
        y: np.ndarray,
        window: Union[float, None] = None,
        itakura_max_slope: Union[float, None] = None,
        bounding_matrix: Union[np.ndarray, None] = None,
        epsilon: float = 1.0,
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size + 1, y_size + 1))

        for i in range(1, x_size + 1):
            for j in range(1, y_size + 1):
                if np.isfinite(bounding_matrix[i - 1, j - 1]):
                    squared_dist = squared_distance(x[:, i - 1], y[:, j - 1])

                    if squared_dist <= epsilon:
                        cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                    else:
                        cost_matrix[i, j] = max(
                            cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                        )

        return cost_matrix[-1, -1], cost_matrix[1:, 1:]

    @staticmethod
    def _result_process(result: float, *args):
        return 1 - float(result / min(args[0].shape[-1], args[1].shape[-1]))

    @staticmethod
    def _alignment_path(
        x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray, bounding_matrix, *args
    ) -> List[Tuple]:
        x = _convert_2d(x)
        y = _convert_2d(y)

        epsilon = 1.0
        if len(args) > 4:
            epsilon = args[4]
        x_size = x.shape[1]
        y_size = y.shape[1]
        i, j = (x_size - 1, y_size - 1)
        path = []

        while i > 0 and j > 0:
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                if squared_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
                    path.append((i - 1, j - 1))
                    i, j = (i - 1, j - 1)
                elif cost_matrix[i - 1][j] > cost_matrix[i][j - 1]:
                    i = i - 1
                else:
                    j = j - 1
        return path[::-1]
