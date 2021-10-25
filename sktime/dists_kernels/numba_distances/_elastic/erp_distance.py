# -*- coding: utf-8 -*-
from typing import Union, Callable
import numpy as np
from numba import njit

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
from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    _numba_euclidean_distance,
)


@njit(cache=True)
def _erp_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
    pre_computed_gx_distances: np.ndarray,
    pre_computed_gy_distances: np.ndarray,
):
    """Method used to calculate the erp cost matrix to derive distance from.

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
    pre_computed_gx_distances: np.ndarray
        pre-computed distances from x to g
    pre_computed_gy_distances: np.ndarray
        pre-computed distances from y to g
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    cost_matrix[1:, 0] = np.sum(pre_computed_gx_distances)
    cost_matrix[0, 1:] = np.sum(pre_computed_gy_distances)
    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + pre_computed_distances[i - 1, j - 1],
                    cost_matrix[i - 1, j] + pre_computed_gx_distances[i - 1],
                    cost_matrix[i, j - 1] + pre_computed_gy_distances[j - 1],
                )
    return cost_matrix[1:, 1:]


@njit(cache=True)
def _numba_erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
    g: float,
) -> float:
    """Method that is a numba compiled version of erp distance.

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

    Returns
    -------
    float
        erp between two timeseries
    """
    _x, _y = _dtw_format_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    pre_computed_gx_distances = _numba_pairwise_distance(
        np.full_like(_x[0], g), _x, True, distance
    )[0]
    pre_computed_gy_distances = _numba_pairwise_distance(
        np.full_like(_y[0], g), _y, True, distance
    )[0]

    cost_matrix = _erp_cost_matrix(
        _x,
        _y,
        bounding_matrix,
        pre_computed_distances,
        pre_computed_gx_distances,
        pre_computed_gy_distances,
    )

    return cost_matrix[-1, -1]


def erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.0,
) -> float:
    """Method to calculate the erp distance between timeseries.

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
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.

    Returns
    -------
    float
        erp distance between the two timeseries
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    bounding_matrix = _resolve_bounding_matrix(
        _x, _y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    return _numba_erp_distance(
        _x, _y, distance, bounding_matrix, np.array_equal(x, y), g
    )


def numba_erp_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool = False,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Method used to produce a numba erp function.

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
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Method that calculates the erp distance between two timeseries
    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_erp(_x: np.ndarray, _y: np.ndarray) -> float:
        return _numba_erp_distance(_y, _x, distance, bounding_matrix, symmetric, g)

    return numba_erp


def pairwise_erp_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    g: float = 0.0,
) -> np.ndarray:
    """Erp pairwise distance between two timeseries.

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
    g: float, defaults = 0.
        Constant that controls the curvature (slope) of the function; that is, g
        controls the level of penalisation for the points with larger phase difference.

    Returns
    -------
    np.ndarray
        Pairwise matrix calculated using erp
    """
    return pairwise_distance(
        x,
        y,
        numba_distance_factory=numba_erp_distance_factory,
        lower_bounding=lower_bounding,
        window=window,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        g=g,
    )
