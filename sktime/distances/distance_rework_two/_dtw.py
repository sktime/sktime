# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()


class _DtwDistance(ElasticDistance):

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
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
        cost_matrix[0, 0] = 0.0

        for i in range(x_size):
            for j in range(y_size):
                if np.isfinite(bounding_matrix[i, j]):
                    squared_dist = squared_distance(x[:, i], y[:, j])
                    cost_matrix[i + 1, j + 1] = squared_dist + min(
                        cost_matrix[i, j + 1],
                        cost_matrix[i + 1, j],
                        cost_matrix[i, j],
                    )

        return cost_matrix[-1, -1], cost_matrix[1:, 1:]
