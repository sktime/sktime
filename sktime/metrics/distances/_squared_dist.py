# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

import numpy as np
from numba import njit, prange
from typing import Callable, Tuple

from sktime.metrics.distances.base.base import BaseDistance, NumbaSupportedDistance


@njit()
def _squared_dist_old(x: np.ndarray, y: np.ndarray) -> float:
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
        for j in prange(dimension_size):
            curr = curr_x[j] - curr_y[j]
            distance += curr * curr

    return distance


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

    distance = 0.0

    for i in prange(x.shape[0]):
        curr = x[i] - y[i]
        distance += np.sum(curr * curr)

    return distance


class SquaredDistance(BaseDistance):
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

    @staticmethod
    def pad_along_axis(
        array: np.ndarray, target_length: int, axis: int = 0
    ) -> np.ndarray:
        """
        Method used to pad a time series along an axis

        Source: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-
                                                        a-tensor-along-some-axis-python

        Parameters
        ----------
        array: np.ndarray
            Array to pad
        target_length: int
            How much to pad by
        axis: int
            Axis to pad along

        Returns
        -------
        np.ndarray
            Padded time series
        """
        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode="constant", constant_values=0)

    def _check_params(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method used to check the parameters of x and y. This is needed in case they
        are not equal length

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        x: np.ndarray
            First padded time series (when needed)
        y: np.ndarray
            Second padded time series (when needed)
        """
        if x.shape[0] == y.shape[0]:
            padded_x = x
            padded_y = y
        elif x.shape[0] > y.shape[0]:
            new_x = x.copy()
            new_y = y.copy()
            padded_x = new_x
            padded_y = self.pad_along_axis(new_y, new_x.shape[0], 0)
        else:
            new_x = x.copy()
            new_y = y.copy()
            padded_x = self.pad_along_axis(new_x, new_y.shape[0], 0)
            padded_y = new_y
        return padded_x, padded_y

    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
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
        padded_x, padded_y = self._check_params(x, y)

        return _squared_dist(padded_x, padded_y)

    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Method used to return a numba callable distance, this assume that all checks
        have been done so the function returned doesn't need to check but checks
        should be done before the return

        The returned numba function also assumes the arrays are padded correctly
        if unequal

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
