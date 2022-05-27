# -*- coding: utf-8 -*-
"""Euclidean distance."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any

import numpy as np
from numba import njit

from sktime.distances.base import DistanceCallable, NumbaDistance


class _EuclideanDistance(NumbaDistance):
    """Euclidean distance between two time series."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled euclidean distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Requires equal length series.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second times eries.
        kwargs: Any
            Extra kwargs. For euclidean there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled euclidean distance callable.
        """
        return _numba_euclidean_distance


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
