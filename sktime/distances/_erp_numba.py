# -*- coding: utf-8 -*-
"""Isolated numba imports for _erp."""

__author__ = ["chrisholder", "TonyBagnall"]

import warnings

import numpy as np

from sktime.utils.numba.njit import njit
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="none"):
    from numba.core.errors import NumbaWarning

    # Warning occurs when using large time series (i.e. 1000x1000)
    warnings.simplefilter("ignore", category=NumbaWarning)


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
