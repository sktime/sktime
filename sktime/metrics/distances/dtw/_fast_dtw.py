# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

import math
from typing import Tuple, Callable, List, Set
import numpy as np
from numba import njit

from sktime.metrics.distances.dtw._lower_bouding import _plot_values_on_matrix
from sktime.metrics.distances.base.base import BaseDistance, NumbaSupportedDistance
from sktime.metrics.distances.dtw._dtw_path import DtwPath


@njit(cache=True)
def _reduce_by_half(x: np.ndarray):
    """
    upper_line_y_values = y.sha[0
    lower_line_y_values = lowe.s[0

    if u!p == lower:
    raise Va("The number of upper line values must equal the number of lower line values

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
    time_series_size = x.shape[1]
    half_x_size = math.floor(x_size / 2)
    half_arr = np.zeros((half_x_size, time_series_size))

    for i in range(half_x_size):
        curr_index = i * 2
        half_arr[i, :] = (x[curr_index] + x[curr_index + 1]) / 2

    return half_arr


@njit(cache=True)
def permutations(
    arr: List,
    arr_len: int,
    x_val: int,
    y_val: int,
    execute_func: Callable[[int, int, int, int], Tuple],
    return_arr: Set,
) -> None:
    """
    Method that is used to calculate the permutations of each value in the array

    Parameters
    ----------
    arr: List
        Array containing the values to calculate permutations for
    arr_len: int
        Integer that is the array length
    x_val: List
        x_val to pass to the execute function
    y_val: List
        y_val to pass to the execute function
    execute_func: Callable[[int, int, int, int], Tuple]
        Function to execute for each permutation
    return_arr: Set
        Set containing the result of each execution of execute_func for each
        permutation

    Returns
    -------
    Set
        Set containing the result of each execution of execute_func for each
        permutation
    """
    for i in range(arr_len):
        curr_i_val = arr[i]
        for j in range(arr_len):
            curr_j_val = arr[j]
            execute_val = execute_func(curr_i_val, curr_j_val, x_val, y_val)
            return_arr.add(execute_val)


@njit(cache=True)
def _path_permutation_func(a: int, b: int, x_val: int, y_val: int) -> Tuple:
    """
    Method used to calculate path permutations

    Parameters
    ----------
    a: int
        First values
    b: int
        Second value
    x_val: int
        X value to evaluate on
    y_val: int
        Y value to evaluate on

    Returns
    -------
    Tuple
        First value is x coordinate, second is y value

    """
    return x_val + a, y_val + b


@njit()
def _calculate_path_values(path: np.ndarray, radius: int) -> Set:
    """
    Method used to calculate the path values

    Parameters
    ----------
    path: List
        Optimal dtw path
    radius: int
        Radius of fast dtw

    Returns
    -------
    Set
        Set containing the path values in range of the radius
    """
    path_permutations = set()
    for i in range(len(path)):
        path_permutations.add((path[i][0], path[i][1]))

    radius_values = list(range(-radius, radius + 1))
    radius_values_len = len(radius_values)

    for i in range(len(path)):
        x_val = path[i][0]
        y_val = path[i][1]

        permutations(
            radius_values,
            radius_values_len,
            x_val,
            y_val,
            _path_permutation_func,
            path_permutations,
        )

    return path_permutations


@njit(cache=True)
def _window_permutation_func(a: int, b: int, x_val: int, y_val: int) -> Tuple:
    """
    Function executed for each window permutation

    Parameters
    ----------
    a: int
        First path coordinate
    b: int
        Second path coordinate
    x_val: int

    y_val

    Returns
    -------

    """
    return x_val * 2 + a, y_val * 2 + b


@njit()
def _calculate_window_values(path_permutations: Set) -> Set:
    """
    Method used to calculate values to go into the bounding matrix for dtw
    Parameters
    ----------
    path_permutations: List
        Different path permutations

    Returns
    -------
    List
        Window values that will make up the bounding matrix
    """
    window_values = set()
    window_values.add((0, 0))

    permutation_values = [0, 1]

    for path_coords in path_permutations:
        permutations(
            permutation_values,
            2,
            path_coords[0],
            path_coords[1],
            _window_permutation_func,
            window_values,
        )

    return window_values


@njit(cache=True)
def _construct_window(window_values, x_size, y_size):
    """
    Method use to construct the values that will make up the bounding matrix for dtw

    Parameters
    ----------
    window_values: List
        List of values to construct the window from
    x_size: int
        Size of x time series
    y_size: int
        Size of y time series

    Returns
    -------
    List
        Window containing the values that will make up the bounding matrix
    """
    # try:
    window = []
    start_j = 0
    for i in range(0, x_size):
        new_start_j = None
        for j in range(start_j, y_size):
            if (i, j) in window_values:
                window.append([i, j])
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    return window


@njit()
def _expand_window(path: np.ndarray, x_size: int, y_size: int, radius: int) -> List:
    """
    Method used to take a reduced path and expand it into a larger one

    Parameters
    ----------
    path: np.ndarray
        Warping path
    x_size: int
        Size of x time series
    y_size: int
        Size of y time series
    radius: int
        Size of the radius to consider for the warping path

    Returns
    -------
    List
        Expanded window to be used for dtw bounding matrix
    """
    path_permutations: Set = _calculate_path_values(path, radius)

    window_values = _calculate_window_values(path_permutations)

    return _construct_window(window_values, x_size, y_size)


@njit()
def _fast_dtw(
    x: np.ndarray,
    y: np.ndarray,
    dist_func: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple],
    radius: int,
) -> Tuple[float, np.ndarray]:
    """
    Recursive method used to calculate fast dtw

    Parameters
    ----------
    x: np.ndarray
        First time series
    y: np.ndarray
        Second time series
    dist_func: Callable[[np.ndarray, np.ndarray, np.ndarray],  Tuple]
        Numba compatible distance function to calculate dtw path
    radius: int
        Radius to calculate fast dtw over

    Returns
    -------
    float
        Distance between time series x and y
    np.ndarray
        Warping path between time series x and y
    """

    min_time_size = radius + 2

    if x.shape[0] < min_time_size or y.shape[0] < min_time_size:
        return dist_func(x, y)

    x_reduce = _reduce_by_half(x)
    y_reduce = _reduce_by_half(y)

    distance, path = _fast_dtw(x_reduce, y_reduce, dist_func, radius)

    window = _expand_window(path, x.shape[0], y.shape[0], radius)

    matrix = np.full((y.shape[0], x.shape[0]), np.inf)

    bounding_matrix = _plot_values_on_matrix(matrix, np.array(window))

    return dist_func(x, y, bounding_matrix)


class FastDtw(BaseDistance, NumbaSupportedDistance):
    """
    Class that defines the FastDtw distance algorithm

    Parameters
    ----------
    radius: int, defaults = 0
        Distance to search outside of the projected warp path from the previous
        resolution when refining the warp path
    """

    def __init__(self, radius=1):
        self.radius = radius

    def _distance(self, x: np.ndarray, y: np.ndarray):
        """
        Method used to compute the distance between two timeseries

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        float
            Distance between time series x and time series y
        """
        radius = self._check_params()
        numba_dtw = DtwPath().numba_distance(x, y)
        distance, _ = _fast_dtw(x, y, numba_dtw, radius)
        # distance, _ = _iterative_dtw(x, y, numba_dtw, radius)
        return distance

    def _check_params(self) -> int:
        """
        Method used to check the parameters for fast dtw

        Returns
        -------
        int
            Radius to use in fast dtw
        """
        if self.radius < 0:
            raise ValueError("Radius must be a positive number")
        return self.radius

    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray, int], float]:
        radius = self._check_params()
        numba_dtw = DtwPath().numba_distance(x, y)

        @njit()
        def numba_fast_dtw(x: np.ndarray, y: np.ndarray, radius: int = radius) -> float:
            distance, _ = _fast_dtw(x, y, numba_dtw, radius)
            return distance

        return numba_fast_dtw
