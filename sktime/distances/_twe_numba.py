"""Isolated numba imports for _twe."""

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
def pad_ts(x: np.ndarray) -> np.ndarray:
    """Pad the time with a 0.0 at the start.

    Parameters
    ----------
    x: np.ndarray (of shape (d, m))
        A time series.

    Returns
    -------
    np.ndarray
        A padded time series of shape (d, m + 1)
    """
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


@njit(cache=True)
def _twe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    lmbda: float,
    nu: float,
    p: int,
) -> np.ndarray:
    """Twe distance compiled to no_python.

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
    lmbda: float
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    nu: float
        A non-negative constant which characterizes the stiffness of the elastic
        TWE measure. Must be > 0.
    p: int
        Order of the p-norm for local cost.

    Returns
    -------
    cost_matrix: np.ndarray (of shape (n, m) where n is the len(x) and m is len(y))
        The dtw cost matrix.
    """
    x = pad_ts(x)
    y = pad_ts(y)
    x_size = x.shape[1]
    y_size = y.shape[1]
    dimensions = x.shape[0]

    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    delete_addition = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                # Deletion in x
                # Euclidean distance to x[:, i - 1] and y[:, i]
                deletion_x_euclid_dist = 0
                for k in range(dimensions):
                    deletion_x_euclid_dist += (x[k][i - 1] - y[k][i]) ** 2
                deletion_x_euclid_dist = np.sqrt(deletion_x_euclid_dist)

                del_x = cost_matrix[i - 1, j] + deletion_x_euclid_dist + delete_addition

                # Deletion in y
                # Euclidean distance to x[:, j - 1] and y[:, j]
                deletion_y_euclid_dist = 0
                for k in range(dimensions):
                    deletion_y_euclid_dist += (x[k][j - 1] - y[k][j]) ** 2
                deletion_y_euclid_dist = np.sqrt(deletion_y_euclid_dist)

                del_y = cost_matrix[i, j - 1] + deletion_y_euclid_dist + delete_addition

                # Keep data points in both time series
                # Euclidean distance to x[:, i] and y[:, j]
                match_same_euclid_dist = 0
                for k in range(dimensions):
                    match_same_euclid_dist += (x[k][i] - y[k][j]) ** 2
                match_same_euclid_dist = np.sqrt(match_same_euclid_dist)

                # Euclidean distance to x[:, i - 1] and y[:, j - 1]
                match_previous_euclid_dist = 0
                for k in range(dimensions):
                    match_previous_euclid_dist += (x[k][i - 1] - y[k][j - 1]) ** 2
                match_previous_euclid_dist = np.sqrt(match_previous_euclid_dist)

                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_euclid_dist
                    + match_previous_euclid_dist
                    + (nu * (2 * abs(i - j)))
                )

                # Choose the operation with the minimal cost and update DP Matrix
                cost_matrix[i, j] = min(del_x, del_y, match)
    return cost_matrix
