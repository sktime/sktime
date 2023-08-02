__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True)
def compute_min_return_path(
    cost_matrix: np.ndarray, bounding_matrix: np.ndarray
) -> List[Tuple]:
    """Compute the minimum cost path through the cost matrix.

    The return path is computed by finding a path through the cost matrix by taking
    the min(cost_matrix[i - 1][j - 1], cost_matrix[i - 1][j], cost_matrix[i][j - 1]).
    This is ideal for dtw based distances or others where the objective is to minimise
    the cost.

    Parameters
    ----------
    cost_matrix: np.ndarray (of size (n, m) where n is the length of the first time
                    series and m is the length of the second time series)
        Cost matrix used to compute the distance.
    bounding_matrix: np.ndarray (of size (n, m) where n is the length of the first
                    time series and m is the length of the second time series)
        The bounding matrix that restricts the warping path.

    Returns
    -------
    list[tuple]
        List containing tuples that is the path through the cost matrix.
    """
    x_size, y_size = cost_matrix.shape

    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i, j] = np.inf

    i = x_size - 1
    j = y_size - 1
    alignment = []
    while True:
        alignment.append((i, j))

        if alignment[-1] == (0, 0):
            break

        arr = np.array(
            [
                cost_matrix[i - 1, j - 1],
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
            ]
        )
        min_index = np.argmin(arr)

        if min_index == 0:
            i = i - 1
            j = j - 1
        elif min_index == 1:
            i = i - 1
        else:
            j = j - 1

    return alignment[::-1]


@njit(cache=True)
def compute_twe_return_path(
    cost_matrix: np.ndarray, bounding_matrix: np.ndarray
) -> List[Tuple]:
    """Compute the twe cost path through the cost matrix.

    The return path is computed by finding a path through the cost matrix by taking
    the min(cost_matrix[i - 1][j - 1], cost_matrix[i - 1][j], cost_matrix[i][j - 1]).
    This is ideal for dtw based distances or others where the objective is to minimise
    the cost.

    Twe is padded with 0s so this is accounted for using this path function.

    Parameters
    ----------
    cost_matrix: np.ndarray (of size (n, m) where n is the length of the first time
                    series and m is the length of the second time series)
        Cost matrix used to compute the distance.
    bounding_matrix: np.ndarray (of size (n, m) where n is the length of the first
                    time series and m is the length of the second time series)
        The bounding matrix that restricts the warping path.

    Returns
    -------
    list[tuple]
        List containing tuples that is the path through the cost matrix.
    """
    x_size, y_size = cost_matrix.shape
    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i, j] = np.inf

    i = cost_matrix.shape[0] - 1
    j = cost_matrix.shape[1] - 1
    alignment = []
    while True:
        alignment.append((i - 1, j - 1))

        if alignment[-1] == (0, 0):
            break

        arr = np.array(
            [
                cost_matrix[i - 1, j - 1],
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
            ]
        )
        min_index = np.argmin(arr)

        if min_index == 0:
            i = i - 1
            j = j - 1
        elif min_index == 1:
            i = i - 1
        else:
            j = j - 1

    return alignment[::-1]


@njit(cache=True)
def compute_lcss_return_path(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounding_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> List[Tuple]:
    """Compute the path from lcss cost matrix.

    Parameters
    ----------
    x: np.ndarray (of shape (dimensions, timepoints)
        First time series.
    y: np.ndarray (of shape (dimensions, timepoints)
        Second time series.
    epsilon : float
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    cost_matrix: np.ndarray (of size (n, m) where n is the length of the first time
                    series and m is the length of the second time series)
        The cost matrix used to compute the lcss distance.
    bounding_matrix: np.ndarray (of size (n, m) where n is the length of the first
                    time series and m is the length of the second time series)
        The bounding matrix that restricts the warping path.

    Returns
    -------
    list[tuple]
        List containing tuples that is the path through the cost matrix.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    dimensions = x.shape[0]

    i, j = (x_size, y_size)
    path = []

    while i > 0 and j > 0:
        if np.isfinite(bounding_matrix[i - 1, j - 1]):
            curr_dist = 0
            for k in range(dimensions):
                curr_dist += (x[k][i - 1] - y[k][j - 1]) ** 2
            curr_dist = np.sqrt(curr_dist)
            if curr_dist <= epsilon:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif cost_matrix[i - 1][j] > cost_matrix[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]
