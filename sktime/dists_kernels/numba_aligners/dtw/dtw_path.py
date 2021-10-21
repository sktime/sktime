# -*- coding: utf-8 -*-
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
def _cost_matrix_to_path(cost_matrix: np.ndarray) -> np.ndarray:
    """Method to turn a cost matrix into a dtw path.

    Parameters
    ----------
    cost_matrix: np.ndarray
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        cost matrix used to calculate the distance.

    Returns
    -------
    np.ndarray
        [n, 2] where n is the number of points in the path. Each row is the index
        of the alignment.
    """
    x_size, x_dim_size = cost_matrix.shape
    start_x = x_size - 1
    start_y = x_dim_size - 1
    path = []

    curr_x = start_x
    curr_y = start_y
    while curr_x != 0 or curr_y != 0:
        path.append([curr_x, curr_y])
        min_coord = np.argmin(
            np.array(
                [
                    cost_matrix[curr_x - 1, curr_y - 1],
                    cost_matrix[curr_x - 1, curr_y],
                    cost_matrix[curr_x, curr_y - 1],
                ]
            )
        )
        if min_coord == 0:
            curr_x -= 1
            curr_y -= 1
        elif min_coord == 1:
            curr_x -= 1
        else:
            curr_y -= 1
    path.append([0, 0])
    return np.array(path[::-1])


# @njit()
def _numba_dtw_path(
    x: np.ndarray,
    y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float],
    bounding_matrix: np.ndarray,
    symmetric: bool,
) -> Tuple[np.ndarray, float]:
    """Method to calculate the dtw path and distance.

    Parameters
    ----------
    x: np.ndarray
        First timeseries
    y: np.ndarray
        Second timeseries
    distance: Callable[[np.ndarray, np.ndarray], float]
    bounding_matrix: np.ndarray
        Numpy matrix containing the bounding matrix with valid cells being finite
        values
    symmetric: bool, defaults = False
        Boolean that is true when the arrays are equal and false when they are not

    Returns
    -------
    np.ndarray
        [n, 2] where n is the number of points in the path. Each row is the index
        of the alignment.
    float
        Distance between the two time series
    """
    _x, _y = _numba_check_params(x, y)

    pre_computed_distances = _numba_pairwise_distance(_x, _y, symmetric, distance)

    cost_matrix = _cost_matrix(_x, _y, bounding_matrix, pre_computed_distances)

    dtw_path = _cost_matrix_to_path(cost_matrix)

    return dtw_path, np.sqrt(cost_matrix[-1, -1])


# def numba_dtw_path_factory(
#         x: np.ndarray,
#         y: np.ndarray,
#         symmetric: bool = False,
#         lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
#         window: int = 2,
#         itakura_max_slope: float = 2.0,
#         distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
#         bounding_matrix: np.ndarray = None,
# ) -> Tuple[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]], int, int]:
#     """Method to create the dtw path alignment numba function.
#
#     Parameters
#     ----------
#     x: np.ndarray
#         First timeseries
#     y: np.ndarray
#         Second timeseries
#     symmetric: bool, defaults = False
#         Boolean that is true when the arrays are equal and false when they are not
#     lower_bounding: LowerBounding or int, defaults = LowerBounding.NO_BOUNDING
#         lower bounding technique to use. Potential bounding techniques and their int
#         value are given below:
#         NO_BOUNDING = 2
#         SAKOE_CHIBA = 2
#         ITAKURA_PARALLELOGRAM = 3
#     window: int, defaults = 2
#         Size of the bounding window
#     itakura_max_slope: float, defaults = 2.
#         Gradient of the slope for itakura
#     defaults = squared_distance
#         Distance function to use
#     distance: Callable[[np.ndarray, np.ndarray], float],
#         defaults = squared_distance
#         Distance function to use within dtw. Defaults to squared distance.
#     bounding_matrix: np.ndarray, defaults = none
#         Custom bounding matrix where inside bounding marked by finite values and
#         outside marked with infinite values.
#
#     Returns
#     -------
#     Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]
#         Callable to get the dtw path and distance
#     int
#         length of x + length of y.
#     int
#         2 as each row is an x and y index for the path.
#     """
#     bounding_matrix = _resolve_bounding_matrix(
#         x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
#     )
#
#     # @njit()
#     def numba_dtw_path(
#             _x: np.ndarray, _y: np.ndarray
#     ) -> Tuple[np.ndarray, float]:
#         return _numba_dtw_path(
#             _y, _x, distance, bounding_matrix, symmetric
#         )
#
#     longer_path = x.shape[0] + y.shape[0] - 1
#
#     return numba_dtw_path, longer_path, 2
#


def dtw_path_alignment(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding: Union[LowerBounding, int] = LowerBounding.NO_BOUNDING,
    window: int = 2,
    itakura_max_slope: float = 2.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = _numba_squared_distance,
    bounding_matrix: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
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
    np.ndarray
        [n, 2] where n is the number of points in the path. Each row is the index
        of the alignment.
    float
        Distance between the two timeseries.
    """
    bounding_matrix = _resolve_bounding_matrix(
        x, y, lower_bounding, window, itakura_max_slope, bounding_matrix
    )

    return _numba_dtw_path(x, y, distance, bounding_matrix, np.array_equal(x, y))
