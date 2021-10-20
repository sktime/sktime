# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
from typing import Callable

from sktime.dists_kernels.numba_distances._utils import to_distance_timeseries


def squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Method used to calculate the squared distance between two series
    Parameters
    ----------
    x: np.ndarray
        First time series
    y: np.ndarray
        Second time series
    Returns
    -------
    distance: float
        squared distance between the two series
    """
    _x = to_distance_timeseries(x)
    _y = to_distance_timeseries(y)
    return _numba_squared_distance(_x, _y)


def numba_squared_distance_factory(
    x: np.ndarray, y: np.ndarray, **kwargs: dict
) -> Callable[[np.ndarray, np.ndarray], float]:
    return _numba_squared_distance


@njit(parallel=True)
def _numba_squared_distance(x, y):
    """Numba implementation of squared distance.

    Parameters
    ----------
    x:
    """
    distance = 0.0
    for i in prange(x.shape[0]):
        curr = x[i] - y[i]
        distance += np.sum(curr * curr)

    return distance
