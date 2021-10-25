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
from sktime.dists_kernels.numba_distances._elastic.euclidean_distance import (
    _numba_euclidean_distance,
)
from sktime.dists_kernels._utils import to_numba_timeseries


@njit(cache=True)
def _sequence_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    pre_computed_distances: np.ndarray,
    epsilon: float,
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
    epsilon : float
        Matching threshold to determine if two subsequences are considered close enough
        to be considered 'common'.

    Returns
    -------
    np.ndarray
        Weighted cost matrix between x and y timeseries
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                if pre_computed_distances[i - 1, j - 1] <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )

    return cost_matrix


@njit(cache=True)
def _numba_lcss_distance(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
    epsilon: float,
) -> float:
    """Method that is a numba compiled version of lcss distance.

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
    epsilon : float
        Matching threshold to determine if two subsequences are considered close enough
        to be considered 'common'.

    Returns
    -------
    float
        lcss between two timeseries
    """
    _x, _y = _dtw_format_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _sequence_cost_matrix(
        _x, _y, bounding_matrix, pre_computed_distances, epsilon
    )
    return float(cost_matrix[-1, -1] / min(_x.shape[0], _y.shape[0]))


def numba_lcss_distance_factory(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool = False,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.SAKOE_CHIBA,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 0.1,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Method used to produce a numba lcss function.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    symmetric: bool, defaults = False
        Boolean that is true when x == y and false when x != y
    lower_bounding: LowerBounding or int, defaults = LowerBounding.SAKOE_CHIBA
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
    epsilon : float, defaults = 0.1
        Matching threshold to determine if two subsequences are considered close enough
        to be considered 'common'.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Method that calculates the lcss distance between two timeseries

    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, delta, itakura_max_slope, bounding_matrix
    )

    @njit()
    def numba_lcss(_x: np.ndarray, _y: np.ndarray) -> float:
        return _numba_lcss_distance(
            _y, _x, distance, bounding_matrix, symmetric, epsilon
        )

    return numba_lcss


def lcss_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.SAKOE_CHIBA,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 0.1,
) -> float:
    """Method to calculate lcss distance between timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    lower_bounding: LowerBounding or int, defaults = LowerBounding.SAKOE_CHIBA
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
    epsilon : float, defaults = 0.1
        Matching threshold to determine if two subsequences are considered close enough
        to be considered 'common'.

    Returns
    -------
    float
        lcss distance between the two timeseries
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    bounding_matrix = _resolve_bounding_matrix(
        _x, _y, lower_bounding, delta, itakura_max_slope, bounding_matrix
    )

    return _numba_lcss_distance(
        _x, _y, distance, bounding_matrix, np.array_equal(x, y), epsilon
    )


def pairwise_lcss_distance(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.SAKOE_CHIBA,
    delta: int = 3,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_euclidean_distance,
    bounding_matrix: np.ndarray = None,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Lcss pairwise distance between two timeseries.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    lower_bounding: LowerBounding or int, defaults = LowerBounding.SAKOE_CHIBA
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
    epsilon : float, defaults = 0.1
        Matching threshold to determine if two subsequences are considered close enough
        to be considered 'common'.

    Returns
    -------
    np.ndarray
        Pairwise matrix calculated using lcss
    """
    return pairwise_distance(
        x,
        y,
        numba_distance_factory=numba_lcss_distance_factory,
        lower_bounding=lower_bounding,
        delta=delta,
        itakura_max_slope=itakura_max_slope,
        distance=distance,
        bounding_matrix=bounding_matrix,
        epsilon=epsilon,
    )
