# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from typing import Callable

from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    _numba_squared_distance,
)
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
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
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    return _numba_euclidean_distance(_x, _y)


def pairwise_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries

    Returns
    -------
    np.ndarray
        Pairwise distance using euclidean distance
    """
    return pairwise_distance(
        x, y, numba_distance_factory=numba_euclidean_distance_factory
    )


def numba_euclidean_distance_factory(
    x: np.ndarray, y: np.ndarray, **kwargs: dict
) -> Callable[[np.ndarray, np.ndarray], float]:
    return _numba_euclidean_distance


@njit()
def _numba_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(_numba_squared_distance(x, y))
