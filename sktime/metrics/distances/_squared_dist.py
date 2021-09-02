# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from typing import Callable

from sktime.metrics.distances.base.base import BaseDistance, NumbaSupportedDistance


@njit()
def _squared_dist(x: np.ndarray, y: np.ndarray) -> float:
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
    x_size = x.shape[0]
    distance = 0.0

    dimension_size = x.shape[1]

    for i in range(x_size):
        curr_x = x[i]
        curr_y = y[i]
        for j in range(dimension_size):
            curr = curr_x[j] - curr_y[j]
            distance += curr * curr

    return distance


class SquaredDistance(BaseDistance, NumbaSupportedDistance):
    """
    Class that is used to calculate the squared distance between a time series. This
    is calculated as follows.

    Given two points x and y the squared distances
        = (x - y)^2

    When x = 2, y = 5
        = (2 - 5)^2
        = 9

    When x = [2, 3, 4], y = [5, 4, 2]
        = (2 - 5)^2 + (3 - 4)^2 + (4 - 2)^2
        = 9 + 1 + 4
        = 14
    """

    def __init__(self):
        super(SquaredDistance, self).__init__("sdistance", {"squared distance"})

    def _distance(self, x: np.ndarray, y: np.ndarray):
        """
        Method used to compute the distance between two ts series
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series
        kwargs: Any
            Key word arguments

        Returns
        -------
        float
            Distance between time series x and time series y
        """
        return _squared_dist(x, y)

    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Method used to return a numba callable distance, this assume that all checks
        have been done so the function returned doesn't need to check but checks
        should be done before the return

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        Callable
            Numba compiled function (i.e. has @njit decorator)
        """

        @njit()
        def _numba_squared_dist(x: np.ndarray, y: np.ndarray) -> float:
            if x.ndim < 2:
                _x = np.reshape(x, (-1, 1))
            else:
                _x = x
            if y.ndim < 2:
                _y = np.reshape(y, (-1, 1))
            else:
                _y = y

            return _squared_dist(_x, _y)

        return _numba_squared_dist
