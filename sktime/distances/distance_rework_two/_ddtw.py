# -*- coding: utf-8 -*-
import numpy as np

from sktime.distances.distance_rework_two._base import (
    ElasticDistance,
    ElasticDistanceReturn,
    get_bounding_matrix,
)
from sktime.distances.distance_rework_two._squared import _SquaredDistance

squared_distance = _SquaredDistance().distance_factory()

def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.

    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q: np.ndarray (of shape (d, m) where d is the dimensions and m is the timepoints.
        A time series.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        Array containing the derivative of q.

    References
    ----------
    .. [1] Keogh E, Pazzani M Derivative dynamic time warping. In: proceedings of 1st
    SIAM International Conference on Data Mining, 2001
    """
    return 0.25 * q[:, 2:] + 0.5 * q[:, 1:-1] - 0.75 * q[:, :-2]


class _DdtwDistance(ElasticDistance):

    _numba_distance = True
    _cache = True
    _fastmath = True

    @staticmethod
    def _preprocess_timeseries(x, *args):
        return average_of_slope(x)

    @staticmethod
    def _distance(
            x: np.ndarray,
            y: np.ndarray,
            window: float = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
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
