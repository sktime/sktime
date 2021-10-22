# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Callable, Tuple
import numpy as np

from sktime.dists_kernels.numba_distances._elastic.dtw.lower_bounding import (
    LowerBounding,
)
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    _numba_squared_distance,
)
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import (
    dtw_distance,
    numba_dtw_distance_factory,
)
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance


def _diff_timeseries(x: np.ndarray):
    """Method to find the difference in timeseries between each element.

    Parameters
    ----------
    x: np.ndarray
        timeseries

    Returns
    -------
    np.ndarray:
        First timeseries with difference between each point calculated
    """
    if x.shape[1] <= 1 and len(x.shape) < 3:
        _x = x.flatten()
        _x = np.diff(_x)
        x_shape = (x.shape[0] - 1, 1)
        _x = np.reshape(_x, x_shape)
    elif len(x.shape) >= 3:
        temp = []
        for val in x:
            temp.append(_diff_timeseries(val))
        _x = np.array(temp)
    else:
        _x = np.diff(x)
    return _x


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
    _x = _diff_timeseries(x)
    _y = _diff_timeseries(y)

    return dtw_distance(
        _x,
        _y,
        lower_bounding,
        window,
        itakura_max_slope,
        distance,
        bounding_matrix,
    )


def _pairwise_ddtw_param_validator(
    x: np.ndarray, y: np.ndarray, **kwargs: dict
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Method to validate and change x and y timeseries to be used for ddtw

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    kwargs: dict
        keyword args

    Returns
    -------
    np.ndarray:
        First timeseries with difference between each point calculated
    np.ndarray:
        Second timeseries with difference between each point calculated
    dict:
        kwargs
    """
    _x = _diff_timeseries(x)
    _y = _diff_timeseries(y)
    return _x, _y, kwargs


def pairwise_ddtw_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ddtw pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries

    Returns
    -------
    np.ndarray
        Pairwise distance using ddtw distance
    """
    return pairwise_distance(
        x,
        y,
        param_validator=_pairwise_ddtw_param_validator,
        numba_distance_factory=numba_dtw_distance_factory,
    )
