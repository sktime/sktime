# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
from numba import njit

from sktime.distances.distance_rework import DistanceCallable, ElasticDistance


class _ErpDistance(ElasticDistance):
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _precompute_g_values(self, dist_func: Callable, g: float, dependent=False):

        if dependent is True:

            def _precompute_g(_x: np.ndarray, _g: float):
                _gx_distance = np.zeros(_x.shape[1])
                _g_arr = np.full(_x.shape[0], _g)
                _x_sum = 0

                for i in range(_x.shape[1]):
                    temp = dist_func(_x[:, i], _g_arr)
                    _gx_distance[i] = temp
                    _x_sum += temp
                return _gx_distance, _x_sum

        else:

            def _precompute_g(_x: np.ndarray, _g: float):
                _x_size = _x.shape[0]
                _gx_distance = np.zeros(_x_size)
                _x_sum = 0

                for i in range(_x_size):
                    temp = dist_func(_x[i], g)
                    _gx_distance[i] = temp
                    _x_sum += temp
                return _gx_distance, _x_sum

        if self._numba_distance is True:
            return njit(cache=self._cache, fastmath=self._fastmath)(_precompute_g)

        return _precompute_g

    def _independent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework import _SquaredDistance

        local_squared_distance = _SquaredDistance().distance_factory(
            x[0], y[0], strategy="local"
        )

        _bounding_matrix = self._get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        _precompute_g_dists = self._precompute_g_values(local_squared_distance, g)

        def _numba_erp(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[0]
            y_size = _y.shape[0]
            cost_matrix = np.zeros((x_size + 1, y_size + 1))

            # Precompute so don't have to do it every iteration
            gx_distance, x_sum = _precompute_g_dists(_x, g)
            gy_distance, y_sum = _precompute_g_dists(_y, g)

            cost_matrix[1:, 0] = x_sum
            cost_matrix[0, 1:] = y_sum

            for i in range(1, x_size + 1):
                for j in range(1, y_size + 1):
                    if np.isfinite(_bounding_matrix[i - 1, j - 1]):
                        squared_dist = local_squared_distance(_x[i - 1], _y[j - 1])

                        cost_matrix[i, j] = min(
                            cost_matrix[i - 1, j - 1] + squared_dist,
                            cost_matrix[i - 1, j] + gx_distance[i - 1],
                            cost_matrix[i, j - 1] + gy_distance[j - 1],
                        )

            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_erp

    def _dependent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework import _SquaredDistance

        _example_x = x[:, 0]
        _example_y = y[:, 0]
        euclidean_distance = _SquaredDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        _bounding_matrix = self._get_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        _precompute_g_dists = self._precompute_g_values(euclidean_distance, g, True)

        def _numba_erp(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[1]
            y_size = _y.shape[1]
            cost_matrix = np.zeros((x_size + 1, y_size + 1))

            # Precompute so don't have to do it every iteration
            gx_distance, x_sum = _precompute_g_dists(_x, g)
            gy_distance, y_sum = _precompute_g_dists(_y, g)

            cost_matrix[1:, 0] = x_sum
            cost_matrix[0, 1:] = y_sum

            for i in range(1, x_size + 1):
                for j in range(1, y_size + 1):
                    if np.isfinite(_bounding_matrix[i - 1, j - 1]):
                        euclidean_dist = euclidean_distance(_x[:, i - 1], _y[:, j - 1])

                        cost_matrix[i, j] = min(
                            cost_matrix[i - 1, j - 1] + euclidean_dist,
                            cost_matrix[i - 1, j] + gx_distance[i - 1],
                            cost_matrix[i, j - 1] + gy_distance[j - 1],
                        )

            return cost_matrix[-1, -1], cost_matrix[1:, 1:]

        return _numba_erp
