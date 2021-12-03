# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import warnings
from typing import Any

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._euclidean import _local_euclidean_distance
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _ErpDistance(NumbaDistance):
    """Edit distance with real penalty (erp) between two timeseries."""

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled erp distance callable.

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
                                        defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        g: float, defaults = 0.
            The reference value to penalise gaps.
        kwargs: Any
            Extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled erp distance callable.

        Raises
        ------
        ValueError
            If the input timeseries is not a numpy array.
            If the input timeseries doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If g is not a float.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if not isinstance(g, float):
            raise ValueError("The value of g must be a float.")

        @njit(cache=True)
        def numba_erp_distance(_x: np.ndarray, _y: np.ndarray) -> float:
            cost_matrix = _erp_cost_matrix(x, y, _bounding_matrix, g)

            return cost_matrix[-1, -1]

        return numba_erp_distance


@njit(cache=True)
def _erp_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
):
    """Compute the erp cost matrix between two timeseries.

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
        The reference value to penalise gaps ('gap' defined when an alignment to
        the next value (in x) in value can't be found).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Erp cost matrix between x and y.
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    x_g = np.full_like(x[0], g)
    y_g = np.full_like(y[0], g)

    gx_distance = np.array([abs(_local_euclidean_distance(x_g, ts)) for ts in x])
    gy_distance = np.array([abs(_local_euclidean_distance(y_g, ts)) for ts in y])
    cost_matrix[1:, 0] = np.sum(gx_distance)
    cost_matrix[0, 1:] = np.sum(gy_distance)

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1]
                    + _local_euclidean_distance(x[i - 1], y[j - 1]),
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )
    return cost_matrix[1:, 1:]
