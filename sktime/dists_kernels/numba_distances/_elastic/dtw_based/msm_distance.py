# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from typing import Union, Callable

from sktime.dists_kernels.numba_distances._elastic.dtw_based.lower_bounding import (
    LowerBounding,
)
from sktime.dists_kernels.numba_distances._elastic.dtw_based.dtw_distance import (
    _dtw_format_params,
    _resolve_bounding_matrix,
)
from sktime.dists_kernels.numba_distances.pairwise_distances import (
    pairwise_distance,
    _numba_pairwise_distance,
)
from sktime.dists_kernels.numba_distances._elastic.absolute_distance import (
    _numba_absolute_distance,
)
from sktime.dists_kernels._utils import to_numba_timeseries


@njit()
def _msm_cost_3d(
    new_point: np.ndarray, x: np.ndarray, y: np.ndarray, c: float
) -> float:
    """Method to derive msm cost from 3d array.

    This is needed due to how numba compiles functions so you have to define different
    for different sized array.

    Parameters
    ----------
    new_point: np.ndarray 3d
        Potential new point.
    x: np.ndarray 3d
        Current x point.
    y: np.ndarray 3d
        Current y point.
    c: float
        Cost value.

    Returns
    -------
    float
        Calculated msm cost.
    """
    cost = 0.0
    for i in range(len(x)):
        cost += _msm_cost_2d(new_point[i], x[i], y[i], c)
    return cost


@njit()
def _msm_cost_2d(
    new_point: np.ndarray, x: np.ndarray, y: np.ndarray, c: float
) -> float:
    """Method to derive msm cost from 2d array.

    This is needed due to how numba compiles functions so you have to define different
    for different sized array.

    Parameters
    ----------
    new_point: np.ndarray 2d
        Potential new point.
    x: np.ndarray 2d
        Current x point.
    y: np.ndarray 2d
        Current y point.
    c: float
        Cost value.

    Returns
    -------
    float
        Calculated msm cost.
    """
    cost = 0.0
    for i in range(len(x)):
        cost += _msm_cost_1d(new_point[i], x[i], y[i], c)
    return cost


@njit()
def _msm_cost_1d(
    new_point: np.ndarray, x: np.ndarray, y: np.ndarray, c: float
) -> float:
    """Method to derive msm cost from 1d array.

    This is needed due to how numba compiles functions so you have to define different
    for different sized array.

    Parameters
    ----------
    new_point: np.ndarray 1d
        Potential new point.
    x: np.ndarray 1d
        Current x point.
    y: np.ndarray 1d
        Current y point.
    c: float
        Cost value.

    Returns
    -------
    float
        Calculated msm cost.
    """
    cost = 0.0
    for i in range(len(x)):
        cost += _msm_cost(new_point[i], x[i], y[i], c)
    return cost


@njit()
def _msm_cost(new_point: float, x: float, y: float, c: float) -> float:
    """Method to calculate msm cost.

    Parameters
    ----------
    new_point: float
        Potential new point.
    x: float
        Current x point.
    y: float
        Current y point.
    c: float
        Cost value.

    Returns
    -------
    float
        Calculated msm cost.
    """
    if ((x <= new_point) and (new_point <= y)) or (
        (y <= new_point) and (new_point <= x)
    ):
        return c
    else:
        return c + min(np.abs(new_point - x), np.abs(new_point - y))


@njit()
def msm_cost(new_point: np.ndarray, x: np.ndarray, y: np.ndarray, c: float) -> float:
    """Method to calculate the msm cost.

    Parameters
    ----------
    new_point: np.ndarray
        Potential new point.
    x: np.ndarray
        Current x point.
    y: np.ndarray
        Current y point.
    c: float
        Cost value.

    Returns
    -------
    float
        Calculated msm cost.
    """
    num_dims = np.shape(x)
    if len(num_dims) == 3:
        msm_cost_func = _msm_cost_3d
    elif len(num_dims) == 2:
        msm_cost_func = _msm_cost_2d
    else:
        msm_cost_func = _msm_cost_1d
    return msm_cost_func(new_point, x, y, c)


@njit()
def _msm_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
    c: float,
):
    """
    Method used to calculate the cost matrix to derive distance from
    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    bounding_matrix: np.ndarray
        Matrix bounding the warping path
    pre_computed_distances: np.ndarray
        Pre-computed distances
    c: float
        Cost value

    Returns
    -------
    np.ndarray
        Weighted cost matrix between x and y timeseries
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size, y_size))

    for i in range(1, x_size):
        cost_matrix[i][0] = cost_matrix[i - 1][0] + msm_cost(x[i], x[i - 1], y[0], c)

    for i in range(1, y_size):
        cost_matrix[0][i] = cost_matrix[0][i - 1] + msm_cost(y[i], x[0], y[i - 1], c)

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + pre_computed_distances[i, j],
                    cost_matrix[i - 1, j] + msm_cost(x[i], x[i - 1], y[j], c),
                    cost_matrix[i, j - 1] + msm_cost(y[j], x[i], y[j - 1], c),
                )
    return cost_matrix[1:, 1:]


@njit()
def _numba_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
    c: float,
) -> float:
    """Method that is a numba compiled version of msm distance.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within wdtw
    bounding_matrix: np.ndarray
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are marked with infinite values.
    symmetric: bool
        Boolean that marks if the two timeseries are the same. If they are then
        true else false. Used to speed up pairwise calculations.
    c: float
        Cost value

    Returns
    -------
    float
        msm between two timeseries
    """
    _x, _y = _dtw_format_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _msm_cost_matrix(_x, _y, bounding_matrix, pre_computed_distances, c)
    return cost_matrix[-1, -1]


def msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_absolute_distance,
    bounding_matrix: np.ndarray = None,
    c: float = 0.1,
) -> float:
    """Method to calculate msm distance between timeseries.

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
    delta: int, defaults = 3
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    c: float, defaults = 0.1
        Cost value

    Returns
    -------
    float
        msm distance between the two timeseries
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    bounding_matrix = _resolve_bounding_matrix(
        _x, _y, lower_bounding, delta, itakura_max_slope, bounding_matrix
    )

    return _numba_msm_distance(
        _x, _y, distance, bounding_matrix, np.array_equal(x, y), c
    )


def numba_msm_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_absolute_distance,
    bounding_matrix: np.ndarray = None,
    c: float = 0.1,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Method used to produce a numba msm function.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    symmetric: bool, defaults = False
        Boolean that is true when x == y and false when x != y
    lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
        lower bounding technique to use. Potential bounding techniques and their int
        value are given below:
        NO_BOUNDING = 2
        SAKOE_CHIBA = 2
        ITAKURA_PARALLELOGRAM = 3
    delta: int, defaults = 3
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    c: float, defaults = 0.1
        Cost value

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Method that calculates the msm distance between two timeseries

    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, delta, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_msm(_x: np.ndarray, _y: np.ndarray) -> float:
        return _numba_msm_distance(_y, _x, distance, bounding_matrix, symmetric, c)

    return numba_msm


def pairwise_msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_absolute_distance,
    bounding_matrix: np.ndarray = None,
    c: float = 0.1,
) -> np.ndarray:
    """Method to calculate msm distance between timeseries.

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
    delta: int, defaults = 3
        Size of the bounding window
    itakura_max_slope: float, defaults = 2.
        Gradient of the slope for itakura
    defaults = squared_distance
        Distance function to use
    distance: Callable[[np.ndarray, np.ndarray], float],
        defaults = squared_distance
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    c: float, defaults = 0.1
        Cost value

    Returns
    -------
    float
        msm distance between the two timeseries
    """

    return pairwise_distance(
        x,
        y,
        numba_distance_factory=numba_msm_distance_factory,
        lower_bounding=lower_bounding,
        delta=delta,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        c=c,
    )
