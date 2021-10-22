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
from sktime.dists_kernels.numba_distances.pairwise_distances import (
    _numba_pairwise_distance,
)
from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba_distances.pairwise_distances import pairwise_distance


def _resolve_bounding_matrix(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    bounding_matrix: np.ndarray = None,
):
    """Method used to resolve the bounding matrix parameters."""
    if bounding_matrix is None:
        if isinstance(lower_bounding, int):
            lower_bounding = LowerBounding(lower_bounding)
        else:
            lower_bounding = lower_bounding

        return lower_bounding.create_bounding_matrix(
            x, y, sakoe_chiba_window_radius=window, itakura_max_slope=itakura_max_slope
        )
    else:
        return bounding_matrix


@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
):
    """
    Method used to calculate the cost matrix to derive distance from
    Parameters
    ----------
    x: np.ndarray
        first timeseries
    y: np.ndarray
        second timeseries
    bounding_matrix: np.ndarray
        matrix bounding the warping path
    pre_computed_distances: np.ndarray
        pre-computed distances
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = pre_computed_distances[i, j] + min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )

    return cost_matrix[1:, 1:]


@njit()
def _numba_check_params(x, y):
    if x.ndim <= 2:
        return np.reshape(x, x.shape + (1,)), np.reshape(y, y.shape + (1,))
    else:
        return x, y


@njit()
def _numba_dtw_distance(x, y, distance, bounding_matrix, symmetric: bool):
    _x, _y = _numba_check_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _cost_matrix(_x, _y, bounding_matrix, pre_computed_distances)
    return np.sqrt(cost_matrix[-1, -1])


def numba_dtw_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool = False,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    symmetric: bool, defaults = False
        Boolean that is true when the arrays are equal and false when they are not
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

    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_dtw(_x: np.ndarray, _y: np.ndarray) -> float:
        return _numba_dtw_distance(_y, _x, distance, bounding_matrix, symmetric)

    return numba_dtw


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> float:
    """Method to calculate dtw distance between timeseries.

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
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.

    Returns
    -------
    float
        dtw distance between the two timeseries
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    bounding_matrix = _resolve_bounding_matrix(
        _x, _y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    return _numba_dtw_distance(_x, _y, distance, bounding_matrix, np.array_equal(x, y))


def pairwise_dtw_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dtw pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries

    Returns
    -------
    np.ndarray
        Pairwise distance using dtw distance
    """
    return pairwise_distance(x, y, numba_distance_factory=numba_dtw_distance_factory)
