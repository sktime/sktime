# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import List, Tuple

import numpy as np
from numba import njit


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

    start = 0
    # Means the cost matrix is padded (i.e. for twe)
    if cost_matrix.shape != bounding_matrix.shape:
        start = 1
    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i - start, j - start]):
                cost_matrix[i, j] = np.inf

    alignment = [(x_size - 1, y_size - 1)]
    while alignment[-1] != (0, 0):
        i, j = alignment[-1]
        if i == 0:
            alignment.append((0, j - 1))
        elif j == 0:
            alignment.append((i - 1, 0))
        else:
            arr = np.array(
                [
                    cost_matrix[i - 1][j - 1],
                    cost_matrix[i - 1][j],
                    cost_matrix[i][j - 1],
                ]
            )

            score = np.argmin(arr)
            if score == 0:
                alignment.append((i - 1, j - 1))
            elif score == 1:
                alignment.append((i - 1, j))
            else:
                alignment.append((i, j - 1))
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
