# -*- coding: utf-8 -*-
"""Euclidean distance."""

__author__ = ["chrisholder"]

from typing import Any

import numpy as np
from numba import njit

from sktime.distances._squared import _numba_squared_distance
from sktime.distances.base import DistanceCallable, NumbaDistance


class _EuclideanDistance(NumbaDistance):
    """Euclidean distance between two timeseries."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled euclidean distance callable.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First timeseries.
        y: np.ndarray (1d or 2d array)
            Second timeseries.
        kwargs: Any
            Extra kwargs. For euclidean there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled euclidean distance callable.
        """
        return _numba_euclidean_distance


@njit(cache=True)
def _numba_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.

    Returns
    -------
    distance: float
        Euclidean distance between x and y.
    """
    distance = _numba_squared_distance(x, y)
    return np.sqrt(distance)
