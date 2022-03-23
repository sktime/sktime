# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "jlines"]

import warnings

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _MsmDistance(NumbaDistance):
    """Move-split-merge (MSM) distance between two timeseries.

    Currently only works with univariate series.
    """

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: float = 0.0,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: dict,
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
        if x.shape[0] > 1 or y.shape[0] > 1:
            raise ValueError(
                f"ERROR, MSM distance currently only works with "
                f"univariate series, passed seris shape {x.shape[0]} and"
                f"shape {y.shape[0]}"
            )
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(cache=True)
        def numba_msm_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _cost_matrix(_x, _y, c, _bounding_matrix)
            return cost_matrix[-1, -1]

        return numba_msm_distance


@njit(cache=True, fastmath=True)
def _dimension_sum(x: np.ndarray, j: int):
    total = 0
    for i in range(x.shape[0]):
        total += x[i][j]

    return total


@njit(cache=True)
def _calc_cost_cell(
    new_point: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: float,
) -> float:
    """Cost calculation function for MSM."""
    new_point_sum = _dimension_sum(new_point)
    x_sum = _dimension_sum(x)
    y_sum = _dimension_sum(y)

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
def _cost_function(x: float, y: float, z: float, c: float) -> float:
    if (y <= x and x <= z) or (y >= x and x >= z):
        return c
    # np.min and abs do not work properly here with numba, no match to floats
    a = x - y
    if a < 0:
        a = -a
    b = x - z
    if b < 0:
        b = -b
    if a > b:
        return c + b
    return c + a


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    bounding_matrix: np.ndarray,
) -> float:
    """MSM distance compiled to no_python.

    Series should be shape (1, m), where m the series (m is currently univariate only).
    length.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).

    Returns
    -------
    distance: float
        MSM distance between the x and y timeseries.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost = np.zeros((x_size, y_size))
    # init the first cell
    if x[0][0] > y[0][0]:
        cost[0, 0] = x[0][0] - y[0][0]
    else:
        cost[0, 0] = y[0][0] - x[0][0]
    # init the rest of the first row and column
    for i in range(1, x_size):
        cost[i][0] = cost[i - 1][0] + _cost_function(x[0][i], x[0][i - 1], y[0][0], c)
    for i in range(1, y_size):
        cost[0][i] = cost[0][i - 1] + _cost_function(y[0][i], y[0][i - 1], x[0][0], c)
    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                d1 = cost[i - 1][j - 1] + np.abs(x[0][i] - y[0][j])
                d2 = cost[i - 1][j] + _cost_function(x[0][i], x[0][i - 1], y[0][j], c)
                d3 = cost[i][j - 1] + _cost_function(y[0][j], x[0][i], y[0][j - 1], c)

            temp = d1
            if d2 < temp:
                temp = d2
            if d3 < temp:
                temp = d3

            cost[i][j] = temp

    return cost[0:, 0:]
