# -*- coding: utf-8 -*-
from typing import Union

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
def _precompute_g(x: np.ndarray, g: float):
    gx_distance = np.zeros(x.shape[1])
    g_arr = np.full(x.shape[0], g)
    x_sum = 0

    for i in range(x.shape[1]):
        temp = squared_distance(x[:, i], g_arr)
        gx_distance[i] = temp
        x_sum += temp
    return gx_distance, x_sum


class _ErpDistance(ElasticDistance):
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
        g: float = 0.0,
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size + 1, y_size + 1))

        # Precompute so don't have to do it every iteration
        gx_distance, x_sum = _precompute_g(x, g)
        gy_distance, y_sum = _precompute_g(y, g)

        cost_matrix[1:, 0] = x_sum
        cost_matrix[0, 1:] = y_sum

        for i in range(1, x_size + 1):
            for j in range(1, y_size + 1):
                if np.isfinite(bounding_matrix[i - 1, j - 1]):
                    squared_dist = squared_distance(x[:, i - 1], y[:, j - 1])

                    cost_matrix[i, j] = min(
                        cost_matrix[i - 1, j - 1] + squared_dist,
                        cost_matrix[i - 1, j] + gx_distance[i - 1],
                        cost_matrix[i, j - 1] + gy_distance[j - 1],
                    )

        return cost_matrix[-1, -1], cost_matrix[1:, 1:]
