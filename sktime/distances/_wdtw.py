# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import warnings
from typing import Any

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._squared import _local_squared_distance
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _WdtwDistance(NumbaDistance):
    """Weighted dynamic time warping (wdtw) distance between two timeseries."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled wdtw distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        window: int, defaults = None
            Integer that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding).
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding).
        bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y)),
                                        defaults = None)
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        g: float, defaults = 0.
            Constant that controls the curvature (slope) of the function; that is, g
            controls the level of penalisation for the points with larger phase
            difference.
        kwargs: Any
            Extra kwargs.


        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled wdtw distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If the value of g is not a float
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if not isinstance(g, float):
            raise ValueError(
                f"The value of g must be a float. The current value is {g}"
            )

        @njit(cache=True)
        def numba_wdtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _weighted_cost_matrix(_x, _y, _bounding_matrix, g)
            return cost_matrix[-1, -1]

        return numba_wdtw_distance


@njit(cache=True)
def _weighted_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
):
    """Compute the wdtw cost matrix between two timeseries.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    g: float
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.

    Returns
    -------
    np.ndarray
        Weighted cost matrix between x and y time series.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    weight_vector = np.array(
        [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
    )

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                ) + weight_vector[np.abs(i - j)] * _local_squared_distance(x[i], y[j])

    return cost_matrix[1:, 1:]
