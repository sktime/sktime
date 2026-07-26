"""Isolated numba imports for _squared."""

__author__ = ["chrisholder", "TonyBagnall"]


import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def _numba_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.

    Returns
    -------
    distance: float
        Squared distance between the x and y.
    """
    dist = 0.0
    for i in range(x.shape[0]):
        dist += _local_squared_distance(x[i], y[i])
    return dist


@njit(cache=True, fastmath=True)
def _local_squared_distance(x: np.ndarray, y: np.ndarray):
    """Compute the local squared distance.

    Parameters
    ----------
    x: np.ndarray (1d array)
        First time series
    y: np.ndarray (1d array)
        Second time series

    Returns
    -------
    float
        Squared distance between the two time series
    """
    distance = 0.0
    for i in range(x.shape[0]):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance
