# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()


class _LcssDistance(ElasticDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _distance(
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
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
