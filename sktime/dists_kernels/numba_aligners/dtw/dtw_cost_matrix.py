# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Callable, Tuple
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
from sktime.dists_kernels.numba_distances._elastic.dtw.dtw_distance import (
    _resolve_bounding_matrix,
    _numba_check_params,
    _cost_matrix,
)


@njit()
def _numba_dtw_cost_matrix_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
) -> Tuple[float, np.ndarray]:
    """Method to calculate the dtw cost matrix and distance.

    Parameters
    ----------
    x: np.ndarray
        First time series
    y: np.ndarray
        Second time series
    distance: Callable[[np.ndarray, np.ndarray], float]
    bounding_matrix: np.ndarray
        Numpy matrix containing the bounding matrix with valid cells being finite
        values
    symmetric: bool, defaults = False
        Boolean that is true when the arrays are equal and false when they are not

    Returns
    -------
    float
        Distance between the two time series
    np.ndarray
        Cost matrix that is [n, m] size where n is the len(x) and m is len(y)
    """
    _x, _y = _numba_check_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _cost_matrix(_x, _y, bounding_matrix, pre_computed_distances)
    return np.sqrt(cost_matrix[-1, -1]), cost_matrix


def numba_dtw_cost_matrix_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool = False,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]:
    """Method to create the cost matrix distance numba function.

    Parameters
    ----------
    x: np.ndarray
        First time series
    y: np.ndarray
        Second time series
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
    Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]
        Callable to get the distance and cost matrix.

    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_dtw(_x: np.ndarray, _y: np.ndarray) -> Tuple[float, np.ndarray]:
        return _numba_dtw_cost_matrix_distance(
            _y, _x, distance, bounding_matrix, symmetric
        )

    return numba_dtw


def dtw_cost_matrix_alignment(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> Tuple[float, np.ndarray]:
    """Method to calculate dtw cost matrix of two timeseries.

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


    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    return _numba_dtw_cost_matrix_distance(
        x, y, distance, bounding_matrix, np.array_equal(x, y)
    )
