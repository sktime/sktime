import warnings
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def compute_return_path(cost_matrix: np.ndarray, bounding_matrix: np.ndarray) -> list:
    """Compute the path from dtw cost matrix.

    Series should be shape (d, m), where d is the number of dimensions, m the series
    length. Series can be different lengths.

    Parameters
    ----------
    cost_matrix: np.ndarray
        Dtw cost matrix to find dtw path through.

    Returns
    -------
    np.ndarray
        Array containing tuple for each path location.
    """
    x_size, y_size = cost_matrix.shape
    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i, j] = np.inf

    alignment = [(x_size - 1, y_size - 1)]
    while alignment[-1] != (0, 0):
        i, j = alignment[-1]
        if i == 0:
            alignment.append((0, j - 1))
        elif j == 0:
            alignment.append((i - 1, 0))
        else:
            arr = np.array([cost_matrix[i - 1][j - 1],
                            cost_matrix[i - 1][j],
                            cost_matrix[i][j - 1]])

            score = np.argmin(arr)
            if score == 0:
                alignment.append((i - 1, j - 1))
            elif score == 1:
                alignment.append((i - 1, j))
            else:
                alignment.append((i, j - 1))
    return alignment[::-1]
