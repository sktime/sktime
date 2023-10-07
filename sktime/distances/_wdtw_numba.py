"""Isolated numba imports for _wdtw."""

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
def _weighted_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
):
    """Compute the wdtw cost matrix between two time series.

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
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.

    Returns
    -------
    np.ndarray
        Weighted cost matrix between x and y time series.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    weight_vector = np.array(
        [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
    )

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                sum = 0
                for k in range(dimensions):
                    sum += (x[k][i] - y[k][j]) * (x[k][i] - y[k][j])
                cost_matrix[i + 1, j + 1] = (
                    min(cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j])
                    + weight_vector[np.abs(i - j)] * sum
                )

    return cost_matrix[1:, 1:]
