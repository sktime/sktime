# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from numba import njit

from sktime.distances.distance_rework._squared_euclidean import _SquaredEuclidean
from sktime.distances.distance_rework.base import BaseDistance, DistanceCostCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix


class _DtwDistance(BaseDistance):
    def _independent_distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict
    ) -> DistanceCostCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        # @njit('Tuple((float64[:, :], float64))(float64[:], float64[:])', cache=True)
        @njit()
        def _dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
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

            return cost_matrix[1:, 1:], cost_matrix[-1, -1]

        return _dtw_distance

    def _dependent_distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict
    ) -> DistanceCostCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        squared_euclidean = _SquaredEuclidean()._independent_distance_factory(
            x[:, 0], y[:, 0], **kwargs
        )

        @njit('Tuple((float64[:, :], float64))(float64[:, :], float64[:, :])',
              cache=True)
        def _dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            x_size = _x.shape[1]
            y_size = _y.shape[1]
            cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
            cost_matrix[0, 0] = 0.0

            for i in range(x_size):
                for j in range(y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        squared_dist = squared_euclidean(_x[:, i], _y[:, j])
                        cost_matrix[i + 1, j + 1] = squared_dist + min(
                            cost_matrix[i, j + 1],
                            cost_matrix[i + 1, j],
                            cost_matrix[i, j],
                        )

            return cost_matrix[1:, 1:], cost_matrix[-1, -1]

        return _dtw_distance
