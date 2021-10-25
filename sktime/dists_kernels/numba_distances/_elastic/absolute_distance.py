# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
from typing import Callable

from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance


def absolute_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Method used to calculate the absolute distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First time series
    y: np.ndarray
        Second time series
    Returns
    -------
    distance: float
        Absolute distance between the two timeseries.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    return _numba_absolute_distance(_x, _y)


def pairwise_absolute_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries

    Returns
    -------
    np.ndarray
        Pairwise distance using absolute distance.
    """
    return pairwise_distance(
        x, y, numba_distance_factory=numba_absolute_distance_factory
    )


def numba_absolute_distance_factory(
    x: np.ndarray, y: np.ndarray, **kwargs: dict
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Method to create a numba compiled absolute distance.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    kwargs: dict
        kwargs for absolute distance.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Numba compiled absolute distance.
    """
    return _numba_absolute_distance


@njit(parallel=True)
def _numba_absolute_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Numba compiled absolute distance.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries

    Returns
    -------
    float
        Absolute distance between x and y.
    """
    distance = 0.0
    for i in prange(x.shape[0]):
        distance += np.sum(x[i] - y[i])

    return np.abs(distance)
