"""Isolated numba imports for _dtw."""

__author__ = ["chrisholder", "TonyBagnall"]

import warnings

import numpy as np

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.numba.njit import njit

if _check_soft_dependencies("numba", severity="none"):
    from numba.core.errors import NumbaWarning

    # Warning occurs when using large time series (i.e. 1000x1000)
    warnings.simplefilter("ignore", category=NumbaWarning)


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    best_known_distance: float = np.inf,
) -> np.ndarray:
    """Dtw distance compiled to no_python.

    Series should be shape (d, m), where d is the number of dimensions, m the series
    length. Series can be different lengths.

    Parameters
    ----------
    x: np.ndarray (2d array of shape dxm1).
        First time series.
    y: np.ndarray (2d array of shape dxm1).
        Second time series.
    bounding_matrix: np.ndarray (2d array of shape m1xm2)
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).
    best_known_distance : float, optional
        Threshold for early abandoning; computation stops if exceeded.    

    Returns
    -------
    cost_matrix: np.ndarray (of shape (n, m) where n is the len(x) and m is len(y))
        The dtw cost matrix.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        min_row_cost = np.inf
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                sum = 0
                for k in range(dimensions):
                    sum += (x[k][i] - y[k][j]) ** 2
                cost_matrix[i + 1, j + 1] = sum
                cost_matrix[i + 1, j + 1] += min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )
                min_row_cost = min(min_row_cost, cost_matrix[i + 1, j + 1])
    # Early abandoning: stop computation if the cumulative cost is too high
    if min_row_cost > best_known_distance:
        return np.full_like(cost_matrix, np.inf)                

    return cost_matrix[1:, 1:]

@njit(cache=True)
def _lb_keogh_envelope(x: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    """Compute the bounding envelope for LB_Keogh lower bound calculation.

    Parameters
    ----------
    x : np.ndarray
        First time series of shape (d, m1).
    y : np.ndarray
        Second time series of shape (d, m2).
    radius : float
        Radius around each time point in `y` to calculate the envelope.

    Returns
    -------
    np.ndarray
        Bounding envelope of `y` for use in LB_Keogh lower bound calculation.
    """
    d, m = x.shape
    n = y.shape[1]
    envelope = np.full((d, m), np.inf)

    for dim in range(d):
        for i in range(m):
            lower_bound = max(0, i - int(radius))
            upper_bound = min(n, i + int(radius) + 1)
            min_value = y[dim, lower_bound]

            # Efficiently compute the minimum in the range
            for j in range(lower_bound + 1, upper_bound):
                if y[dim, j] < min_value:
                    min_value = y[dim, j]

            envelope[dim, i] = min_value

    return envelope


@njit(cache=True)
def _lb_keogh_distance(x: np.ndarray, y: np.ndarray, envelope: np.ndarray) -> float:
    """Compute the LB_Keogh lower bound distance.

    Parameters
    ----------
    x : np.ndarray
        First time series of shape (d, m1).
    y : np.ndarray
        Second time series of shape (d, m2).
    envelope : np.ndarray
        Precomputed bounding envelope of the second time series.

    Returns
    -------
    float
        Lower bound distance between `x` and `y`.
    """
    dist = 0.0
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            if x[j, i] > envelope[j, i]:
                dist += (x[j, i] - envelope[j, i]) ** 2
    return np.sqrt(dist)