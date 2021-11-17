# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Any

import numpy as np
from numba import njit

from sktime.distances.base import DistanceCallable, NumbaDistance


class _SquaredDistance(NumbaDistance):
    """Squared distance between two timeseries."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled Squared distance callable.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First timeseries.
        y: np.ndarray (1d or 2d array)
            Second timeseries.
        kwargs: Any
            Extra kwargs. For squared there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Squared distance callable.
        """
        return _numba_squared_distance


@njit(cache=True)
def _numba_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance compiled to no_python.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.

    Returns
    -------
    distance: float
        Squared distance between the x and y.

    """
    dist = 0.0
    x_size = x.shape[0]
    y_size = y.shape[0]
    min_dist = min(x_size, y_size)
    for i in range(min_dist):
        curr_x = x[i]
        curr_y = y[i]
        diff = curr_x - curr_y
        # diff = np.square(diff)
        for curr_diff in diff:
            dist += curr_diff * curr_diff
            # dist += curr_diff
    return dist
