import numpy as np
from numba import njit
from typing import Callable, Tuple, Union

from sktime.distances.distance_rework.base import BaseDistance, DistanceCostCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix
from sktime.distances.distance_rework._squared_euclidean import _SquaredEuclidean


class _WdtwDistance(BaseDistance):
    def _independent_distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: int = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            g: float = 0.05,
            **kwargs: dict
    ) -> DistanceCostCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(cache=True)
        def _wdtw_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            x_size = x.shape[1]
            y_size = y.shape[1]
            cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
            cost_matrix[0, 0] = 0.0

            weight_vector = np.array(
                [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
            )

            for i in range(x_size):
                for j in range(y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        squared_dist = (_x[i] - _y[j]) ** 2
                        cost_matrix[i + 1, j + 1] = \
                            squared_dist * weight_vector[i - j] + min(
                                    cost_matrix[i, j + 1],
                                    cost_matrix[i + 1, j],
                                    cost_matrix[i, j]
                                )

            return cost_matrix[1:, 1:], cost_matrix[-1, -1]
        return _wdtw_distance

    def _dependent_distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: int = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            g: float = 0.05,
            **kwargs: dict
    ) -> DistanceCostCallable:
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        squared_euclidean = _SquaredEuclidean()._independent_distance_factory(
            x[:, 0], y[:, 0], **kwargs
        )

        @njit(cache=True)
        def _wdtw_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            x_size = x.shape[1]
            y_size = y.shape[1]
            cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
            cost_matrix[0, 0] = 0.0

            weight_vector = np.array(
                [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
            )

            for i in range(x_size):
                for j in range(y_size):
                    if np.isfinite(_bounding_matrix[i, j]):
                        squared_dist = squared_euclidean(_x[:, i], _y[:, j])
                        cost_matrix[i + 1, j + 1] = \
                            squared_dist * weight_vector[i - j] + min(
                                cost_matrix[i, j + 1],
                                cost_matrix[i + 1, j],
                                cost_matrix[i, j]
                            )


            return cost_matrix[1:, 1:], cost_matrix[-1, -1]

        return _wdtw_distance
