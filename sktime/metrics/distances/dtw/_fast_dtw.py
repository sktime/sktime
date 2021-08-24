# -*- coding: utf-8 -*-
import math
from typing import Union, Tuple, Callable
import numpy as np
from numba import njit, prange

from sktime.metrics.distances.dtw._lower_bouding import LowerBounding
from sktime.metrics.distances.base.base import BaseDistance, BasePairwise
from sktime.metrics.distances._squared_dist import SquaredDistance
from sktime.utils.numba_utils import np_mean


def coarsening(self, x, y):
    pass


@njit(parallel=True)
def _reduce_by_half(x: np.ndarray):
    """
    Method used to reduce the size of a given time series. This is done by averaging
    adjacent points in the series
    Parameters
    ----------
    x: np.ndarray
        Time series to reduce by half

    Returns
    -------
    np.ndarray
        Time series that is reduced by half
    """
    x_size = x.shape[0]
    dim_size = x.shape[1]
    half_x_size = math.floor(dim_size / 2)
    half_arr = np.zeros((x_size, half_x_size))
    for i in prange(x_size):
        num_points = dim_size
        if dim_size % 2 != 0:
            num_points -= 1

        # inside njit np.mean doesn't work so calling a utils method to replace it
        half_arr[i, :] = np_mean(np.reshape(x[i][:num_points], (-1, 2)))

    return half_arr


def projection(self, x, y):
    pass


def refinment(self, x, y):
    pass


def _fast_dtw(x, y, radius, dist):
    min_time_size = radius + 2

    while x.shape[1] >= min_time_size or y.shape[1] >= min_time_size:
        x = _reduce_by_half(x)
        y = _reduce_by_half(y)

    distance, path = _fast_dtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    # window = _expand_window(path, len(x), len(y), radius)
    return
    # return __dtw(x, y, window, dist=dist)


class FastDtw(BaseDistance, BasePairwise):
    """
    Class that defines the FastDtw distance algorithm

    Parameters
    ----------
    radius: int, defaults = 2
        Distance to search outside of the projected warp path from the previous
        resolution when refining the warp path
    """

    def __init__(self, radius=2):
        super(FastDtw, self).__init__("fastdtw", {"fast dynamic me warping"})
        self.radius = radius

    def _distance(self, x: np.ndarray, y: np.ndarray):
        test = _reduce_by_half(x)
        pass

    def _pairwise(self, x: np.ndarray, y: np.ndarray, symmetric: bool) -> np.ndarray:
        pass
