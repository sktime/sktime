"""Isolated numba imports for _euclidean."""

__author__ = ["chrisholder", "TonyBagnall"]

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def _local_euclidean_distance(x, y):
    """Compute the local euclidean distance.

    Parameters
    ----------
    x: np.ndarray (1d array)
        First time series
    y: np.ndarray (1d array)
        Second time series

    Returns
    -------
    float
        Euclidean distance between the two time series
    """
    distance = 0.0
    for i in range(x.shape[0]):
        difference = x[i] - y[i]
        distance += difference * difference

    return np.sqrt(distance)


@njit(cache=True, fastmath=True)
def _numba_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array shape (d,m))
        First time series.
    y: np.ndarray (2d array shape (d,m))
        Second time series.

    Returns
    -------
    distance: float
        Euclidean distance between x and y.
    """
    distance = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            difference = x[i][j] - y[i][j]
            distance += difference * difference
    return np.sqrt(distance)
