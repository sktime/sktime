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
from sktime.dists_kernels.numba_distances._elastic.squared_distance import (
    _numba_squared_distance,
)
from sktime.dists_kernels._utils import to_numba_timeseries


@njit(cache=True)
def _weighted_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
    weight_vector: np.ndarray,
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
    weight_vector: np.ndarray
        Array that contains weights for each element

    Returns
    -------
    np.ndarray
        Weighted cost matrix between x and y timeseries
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i + 1, j + 1] = (
                    min(cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j])
                    + weight_vector[np.abs(i - j)] * pre_computed_distances[i, j]
                )

    return cost_matrix[1:, 1:]


@njit()
def _numba_wdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
    g: float,
) -> float:
    """Method that a numba compiled distance for wdtw.

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
    g: float
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalisation is towards
        points.

    Returns
    -------
    float
        Wdtw between two timeseries
    """
    _x, _y = _dtw_format_params(x, y)
    x_size = _x.shape[0]

    weight_vector = np.array(
        [1 / (1 + np.exp(-g * (i - x_size / 2))) for i in range(0, x_size)]
    )

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _weighted_cost_matrix(
        _x, _y, bounding_matrix, pre_computed_distances, weight_vector
    )
    return cost_matrix[-1, -1]


def numba_wdtw_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool = False,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.05,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Method used to produce a numba wdtw function.

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
        Distance function to use within wdtw. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    g: float, defaults = 0.05
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalision is towards
        points.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Method that calculates the wdtw distance between two timeseries

    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_wdtw(_x: np.ndarray, _y: np.ndarray) -> float:
        return _numba_wdtw_distance(_y, _x, distance, bounding_matrix, symmetric, g)

    return numba_wdtw


def wdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.05,
) -> float:
    """Method to calculate wdtw distance between timeseries.

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
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    g: float, defaults = 0.05
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalision is towards
        points.

    Returns
    -------
    float
        wdtw distance between the two timeseries
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    bounding_matrix = _resolve_bounding_matrix(
        _x, _y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    return _numba_wdtw_distance(
        _x, _y, distance, bounding_matrix, np.array_equal(x, y), g
    )


def pairwise_wdtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.05,
) -> np.ndarray:
    """Wdtw pairwise distance between two timeseries.

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
        Distance function to use within dtw_based. Defaults to squared distance.
    bounding_matrix: np.ndarray, defaults = none
        Custom bounding matrix where inside bounding marked by finite values and
        outside marked with infinite values.
    g: float, defaults = 0.05
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.
        tldr: it controls how aggressive the weighting reward or penalision is towards
        points.

    Returns
    -------
    np.ndarray
        Pairwise matrix calculated using wdtw
    """
    return pairwise_distance(
        x,
        y,
        numba_distance_factory=numba_wdtw_distance_factory,
        lower_bounding=lower_bounding,
        window=window,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        g=g,
    )
