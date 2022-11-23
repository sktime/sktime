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


@njit(fastmath=True, cache=True)
def _cost_dependent(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
    diameter = squared_distance(y, z)
    mid = (y + z) / 2
    distance_to_mid = squared_distance(mid, x)

    if distance_to_mid <= (diameter / 2):
        return c
    else:
        dist_to_q_prev = squared_distance(y, x)
        dist_to_c = squared_distance(z, x)
        if dist_to_q_prev < dist_to_c:
            return c + dist_to_q_prev
        else:
            return c + dist_to_c


@njit(fastmath=True, cache=True)
def _cost_independent(x: float, y: float, z: float, c: float) -> float:
    if (y <= x <= z) or (y >= x >= z):
        return c
    return c + min(abs(x - y), abs(x - z))


class _MsmDistance(ElasticDistance):

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
        c: float = 1.0,
        *args
    ) -> ElasticDistanceReturn:
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size, y_size))
        cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

        for i in range(1, x_size):
            cost = _cost_dependent(x[:, i], x[:, i - 1], y[:, 0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
        for i in range(1, y_size):
            cost = _cost_dependent(y[:, i], y[:, i - 1], x[:, 0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

        for i in range(1, x_size):
            for j in range(1, y_size):
                if np.isfinite(bounding_matrix[i, j]):
                    d1 = cost_matrix[i - 1][j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                    d2 = cost_matrix[i - 1][j] + _cost_dependent(
                        x[:, i], x[:, i - 1], y[:, j], c
                    )
                    d3 = cost_matrix[i][j - 1] + _cost_dependent(
                        y[:, j], x[:, i], y[:, j - 1], c
                    )

                    cost_matrix[i, j] = min(d1, d2, d3)

        return cost_matrix[-1, -1], cost_matrix

    @staticmethod
    def _independent_distance(
        y: np.ndarray,
        x: np.ndarray,
        window: Union[float, None] = None,
        itakura_max_slope: Union[float, None] = None,
        bounding_matrix: Union[np.ndarray, None] = None,
        c: float = 1.0,
        *args
    ) -> float:
        """Independent msm distance between two time series.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        """
        x_size = x.shape[1]
        y_size = y.shape[1]
        bounding_matrix = get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        cost_matrix = np.zeros((x_size, y_size))
        cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

        for i in range(1, x_size):
            cost = _cost_independent(x[0, i], x[0, i - 1], y[0, 0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
        for i in range(1, y_size):
            cost = _cost_independent(y[0, i], y[0, i - 1], x[0, 0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

        for i in range(1, x_size):
            for j in range(1, y_size):
                if np.isfinite(bounding_matrix[i, j]):
                    d1 = cost_matrix[i - 1][j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                    d2 = cost_matrix[i - 1][j] + _cost_independent(
                        x[0, i], x[0, i - 1], y[0, j], c
                    )
                    d3 = cost_matrix[i][j - 1] + _cost_independent(
                        y[0, j], x[0, i], y[0, j - 1], c
                    )

                    cost_matrix[i, j] = min(d1, d2, d3)

        return cost_matrix[-1, -1], cost_matrix
