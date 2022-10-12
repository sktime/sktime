# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

import warnings
from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._distance_alignment_paths import compute_min_return_path
from sktime.distances.base import (
    DistanceAlignmentPathCallable,
    DistanceCallable,
    NumbaDistance,
)
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _ErpDistance(NumbaDistance):
    """Edit distance with real penalty (erp) between two time series."""

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs: Any
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled erp distance alignment path callable.

        Similar to LCSS with a different penalty.
        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
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
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
            No_python compiled wdtw distance path callable.

        Raises
        ------
        ValueError
            If the input times eries is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
            If g is not a float.
        """
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )
        if not isinstance(g, float):
            raise ValueError("The value of g must be a float.")

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_erp_distance_alignment_path(
                _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[List, float, np.ndarray]:
                cost_matrix = _erp_cost_matrix(_x, _y, _bounding_matrix, g)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_erp_distance_alignment_path(
                _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[List, float]:
                cost_matrix = _erp_cost_matrix(_x, _y, _bounding_matrix, g)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1]

        return numba_erp_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
        **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled erp distance callable.

        Similar to LCSS with a different penalty.
        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
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
            If the input time series is not a numpy array.
            If the input time series doesn't have exactly 2 dimensions.
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
            cost_matrix = _erp_cost_matrix(_x, _y, _bounding_matrix, g)

            return cost_matrix[-1, -1]

        return numba_erp_distance


@njit(cache=True)
def _erp_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
):
    """Compute the erp cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
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
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))
    gx_distance = np.zeros(x_size)
    gy_distance = np.zeros(y_size)
    for j in range(x_size):
        for i in range(dimensions):
            gx_distance[j] += (x[i][j] - g) * (x[i][j] - g)
        gx_distance[j] = np.sqrt(gx_distance[j])
    for j in range(y_size):
        for i in range(dimensions):
            gy_distance[j] += (y[i][j] - g) * (y[i][j] - g)
        gy_distance[j] = np.sqrt(gy_distance[j])
    cost_matrix[1:, 0] = np.sum(gx_distance)
    cost_matrix[0, 1:] = np.sum(gy_distance)

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                curr_dist = 0
                for k in range(dimensions):
                    curr_dist += (x[k][i - 1] - y[k][j - 1]) * (
                        x[k][i - 1] - y[k][j - 1]
                    )
                curr_dist = np.sqrt(curr_dist)
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + curr_dist,
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )
    return cost_matrix[1:, 1:]
