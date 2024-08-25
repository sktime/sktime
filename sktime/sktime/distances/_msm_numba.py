"""Isolated numba imports for _msm."""

__author__ = ["chrisholder", "jlines", "TonyBagnall"]

import warnings

import numpy as np

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.numba.njit import njit

if _check_soft_dependencies("numba", severity="none"):
    from numba.core.errors import NumbaWarning

    # Warning occurs when using large time series (i.e. 1000x1000)
    warnings.simplefilter("ignore", category=NumbaWarning)


@njit(fastmath=True, cache=True)
def _cost(_x: float, _y: float, _z: float, _c: float) -> float:
    if (_y <= _x <= _z) or (_y >= _x >= _z):
        return _c
    return _c + min(abs(_x - _y), abs(_x - _z))


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    c: float,
) -> np.ndarray:
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
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Erp cost matrix between x and y.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    # init the first cell
    if x[0][0] > y[0][0]:
        cost_matrix[0, 0] = x[0][0] - y[0][0]
    else:
        cost_matrix[0, 0] = y[0][0] - x[0][0]
    # init the rest of the first row and column
    for i in range(1, x_size):
        cost = _cost(x[0, i], x[0, i - 1], y[0, 0], c)
        cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
    for i in range(1, y_size):
        cost = _cost(y[0, i], y[0, i - 1], x[0, 0], c)
        cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                d1 = cost_matrix[i - 1, j - 1] + np.abs(x[0][i] - y[0][j])
                d2 = cost_matrix[i - 1, j] + _cost(x[0][i], x[0][i - 1], y[0][j], c)
                d3 = cost_matrix[i, j - 1] + _cost(y[0][j], x[0][i], y[0][j - 1], c)
                cost_matrix[i][j] = min(d1, d2, d3)
    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                d1 = cost_matrix[i - 1, j - 1] + abs(x[0, i] - y[0, j])
                d2 = cost_matrix[i - 1, j] + _cost(x[0, i], x[0, i - 1], y[0, j], c)
                d3 = cost_matrix[i, j - 1] + _cost(y[0, j], x[0, i], y[0, j - 1], c)
                cost_matrix[i, j] = min(d1, d2, d3)

    return cost_matrix
