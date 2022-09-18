# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from sktime.distances.distance_rework import BaseDistance, DistanceCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix


class _MsmDistance(BaseDistance):
    _has_cost_matrix = True
    _numba_distance = True
    _cache = True
    _fastmath = True

    def _independent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        c: float = 1.0,
        **kwargs: dict
    ) -> DistanceCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(fastmath=self._fastmath, cache=self._cache)
        def _cost(_x: float, _y: float, _z: float, _c: float) -> float:
            if (_y <= _x <= _z) or (_y >= _x >= _z):
                return _c
            return _c + min(abs(_x - _y), abs(_x - _z))

        def _numba_msm(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = _x.shape[0]
            y_size = _y.shape[0]
            cost_matrix = np.zeros((x_size, y_size))

            cost_matrix[0, 0] = abs(_x[0] - _y[0])

            # init the rest of the first row and column
            for i in range(1, x_size):
                cost = _cost(_x[i], _x[i - 1], _y[0], c)
                cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
            for i in range(1, y_size):
                cost = _cost(_y[i], _y[i - 1], _x[0], c)
                cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

            for i in range(1, x_size):
                for j in range(1, y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        d1 = cost_matrix[i - 1, j - 1] + abs(_x[i] - _y[j])
                        d2 = cost_matrix[i - 1, j] + _cost(_x[i], _x[i - 1], _y[j], c)
                        d3 = cost_matrix[i, j - 1] + _cost(_y[j], _x[i], _y[j - 1], c)
                        cost_matrix[i, j] = min(d1, d2, d3)
            return cost_matrix[-1, -1], cost_matrix

        return _numba_msm

    def _dependent_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        c: float = 1.0,
        **kwargs: dict
    ) -> DistanceCallable:
        # Has to be here because circular import if at top
        from sktime.distances.distance_rework import _EuclideanDistance

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        _example_x = x[:, 0]
        _example_y = y[:, 0]
        euclidean_distance = _EuclideanDistance().distance_factory(
            _example_x, _example_y, strategy="independent", **kwargs
        )

        @njit(fastmath=self._fastmath, cache=self._cache)
        def _cost(_x: np.ndarray, _y: np.ndarray, _z: np.ndarray, _c: float) -> float:
            diameter = euclidean_distance(_y, _z)
            mid = (_y + _z) / 2
            distance_to_mid = euclidean_distance(mid, _x)

            if distance_to_mid <= (diameter / 2):
                return c
            else:
                dist_to_q_prev = euclidean_distance(_y, _x)
                dist_to_c = euclidean_distance(_z, _x)
                if dist_to_q_prev < dist_to_c:
                    return c + dist_to_q_prev
                else:
                    return c + dist_to_c

        def _numba_msm(
            _x: np.ndarray,
            _y: np.ndarray,
        ):
            x_size = x.shape[1]
            y_size = y.shape[1]
            cost_matrix = np.zeros((x_size, y_size))
            cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

            for i in range(1, x_size):
                cost = _cost(_x[:, i], _x[:, i - 1], _y[:, 0], c)
                cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
            for i in range(1, y_size):
                cost = _cost(_y[:, i], _y[:, i - 1], _x[:, 0], c)
                cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

            for i in range(1, x_size):
                for j in range(1, y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        d1 = cost_matrix[i - 1][j - 1] + np.sum(
                            np.abs(_x[:, i] - _y[:, j])
                        )
                        d2 = cost_matrix[i - 1][j] + _cost(
                            _x[:, i], _x[:, i - 1], _y[:, j], c
                        )
                        d3 = cost_matrix[i][j - 1] + _cost(
                            _y[:, j], _x[:, i], _y[:, j - 1], c
                        )

                        cost_matrix[i, j] = min(d1, d2, d3)

            return cost_matrix[-1, -1], cost_matrix

        return _numba_msm
