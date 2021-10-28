# -*- coding: utf-8 -*-
"""Euclidean distance."""

__author__ = ["chrisholder"]

import numpy as np
from numba import njit

from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance


class _EuclideanDistance(NumbaDistance):
    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCallable:
        """Create a no_python compiled Euclidean distance callable.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.
        kwargs: dict
            Extra kwargs. For euclidean there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Euclidean distance callable.
        """
        return _EuclideanDistance._numba_distance

    @staticmethod
    @njit(cache=True)
    def _numba_distance(x, y) -> float:
        """Euclidean distance compiled to numba.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First timeseries.
        y: np.ndarray (2d array)
            Second timeseries.

        Returns
        -------
        distance: float
            Euclidean distance between the two timeseries.
        """
        distance = 0.0
        for i in range(x.shape[0]):
            curr = x[i] - y[i]
            distance += np.sum(np.sqrt(curr * curr))

        return distance
