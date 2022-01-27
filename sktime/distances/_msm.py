# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "jlines"]

import warnings

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances.base import DistanceCallable, NumbaDistance

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _MsmDistance(NumbaDistance):
    """Move-split-merge (MSM) distance between two timeseries."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, c: float = 0.0, **kwargs: dict
    ) -> DistanceCallable:
        """Create a no_python compiled MSM distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        c: float
            parameter used in MSM (update later!)

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled MSM distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
        """

        @njit(cache=True)
        def numba_msm_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _cost_matrix(_x, _y, c)
            return cost_matrix[-1, -1]

        return numba_msm_distance


@njit(cache=True, fastmath=True)
def _sum_arr(x):
    total = 0
    for i in range(x.shape[0]):
        total += x[i]

    return total


@njit(cache=True)
def _calc_cost_cell(
    new_point: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: float,
) -> float:
    """Cost calculation function for MSM."""
    new_point_sum = _sum_arr(new_point)
    x_sum = _sum_arr(x)
    y_sum = _sum_arr(y)

    if ((x_sum <= new_point_sum) and (new_point_sum <= y_sum)) or (
        (y_sum <= new_point_sum) and new_point_sum <= x_sum
    ):
        return c
    else:
        a = np.abs(new_point_sum - x_sum)
        b = np.abs(new_point_sum - y_sum)

        if a < b:
            return c + a
        else:
            return c + b
        # return c + np.min([np.abs(new_point - x), np.abs(new_point - y)])


@njit(cache=True)
def _cost_matrix(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """MSM distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).

    Returns
    -------
    distance: float
        MSM distance between the x and y timeseries.
    """
    m = x.shape[0]
    n = y.shape[0]
    cost = np.zeros((m, n))

    # init the first cell
    cost[0, 0] = np.abs(_sum_arr(x[0]) - _sum_arr(y[0]))

    # init the rest of the first row and column
    for i in range(1, m):
        cost[i][0] = cost[i - 1][0] + _calc_cost_cell(x[i], x[i - 1], y[0], c)
    for i in range(1, n):
        cost[0][i] = cost[0][i - 1] + _calc_cost_cell(y[i], y[i - 1], x[0], c)

    for i in range(1, m):
        for j in range(1, n):
            d1 = np.sum(cost[i - 1][j - 1] + np.abs(x[i] - y[j]))
            d2 = cost[i - 1][j] + _calc_cost_cell(x[i], x[i - 1], y[j], c)
            d3 = cost[i][j - 1] + _calc_cost_cell(y[j], x[i], y[j - 1], c)

            temp = d1
            if d2 < temp:
                temp = d2
            if d3 < temp:
                temp = d3

            cost[i][j] = temp

    return cost[0:, 0:]
