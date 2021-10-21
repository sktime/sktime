# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Callable
import numpy as np
from numba import njit

from sktime.dists_kernels.numba_distances._elastic.dtw.lower_bounding import (
    LowerBounding,
)
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    _numba_squared_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import dtw_distance


def ddtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> float:
    """Method to calculate ddtw distance between timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        lower bounding technique to use. Potential bounding techniques and their int
        value are given below:
        NO_BOUNDING = 2
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    window: int, defaults = 2
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.

    Returns
    -------
    float
        ddtw distance between the two timeseries
    """
    _x = x
    _y = y
    if x.shape[1] <= 1 and len(x.shape) < 3:
        _x = x.flatten()

    if y.shape[1] <= 1 and len(y.shape) < 3:
        _y = y.flatten()

    return dtw_distance(
        np.diff(_x),
        np.diff(_y),
        lower_bounding,
        window,
        itakura_max_slope,
        distance,
        bounding_matrix,
    )
