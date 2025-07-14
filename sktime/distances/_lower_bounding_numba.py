"""Isolated numba imports for lower_bounding."""

__author__ = ["chrisholder", "TonyBagnall"]

import math
from typing import Union

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True)
def create_shape_on_matrix(
    bounding_matrix: np.ndarray,
    y_upper_line: np.ndarray,
    y_lower_line: Union[np.ndarray, None] = None,
    x_step_size: int = 1,
    start_val: int = 0,
) -> np.ndarray:
    """Create a shape from a given upper line and lower line on a matrix.

    Parameters
    ----------
    bounding_matrix: np.ndarray (2d array)
        Matrix of size mxn where m is len(x) and n is len(y). Values that
        are inside the shape will be replaced with finite values (0.).
    y_upper_line: np.ndarray (1d array)
        Y points of the upper line.
    y_lower_line: np.ndarray (1d array), defaults = None
        Y points of the lower line. If no lower line specified, then y_upper_line
        used as lower line.
    x_step_size: int, defaults = 1
        Step size each iteration will increase by
    start_val: int, defaults = 0
        Starting coordinate for x

    Returns
    -------
    np.ndarray (2d array)
        Matrix with values of the shape set to 0. (finite), of the same shape
        as the passed bounding_matrix.
    """
    y_size = bounding_matrix.shape[0]

    if y_lower_line is None:
        y_lower_line = y_upper_line

    upper_line_y_values = y_upper_line.shape[0]
    lower_line_y_values = y_lower_line.shape[0]

    if upper_line_y_values != lower_line_y_values:
        raise ValueError(
            "The number of upper line values must equal the number of lower line values"
        )

    for i in range(start_val, upper_line_y_values):
        x = i * x_step_size

        upper_y = max(0, min(y_size - 1, math.ceil(y_upper_line[i])))
        lower_y = max(0, min(y_size - 1, math.floor(y_lower_line[i])))

        if upper_y == lower_y:
            bounding_matrix[upper_y, x] = 0.0
        else:
            bounding_matrix[upper_y : (lower_y + 1), x] = 0.0

    return bounding_matrix


@njit(cache=True)
def _check_line_steps(line: np.ndarray) -> np.ndarray:
    """Check the next 'step' is along the line.

    Parameters
    ----------
    line: np.ndarray
        line to check steps.

    Returns
    -------
    np.ndarray
        Line with updated indexes.
    """
    prev = line[0]
    for i in range(1, len(line)):
        curr_val = line[i]
        if curr_val > (prev + 1):
            line[i] = prev + 1
        elif curr_val < (prev - 1):
            line[i] = prev - 1
        prev = curr_val
    return line


@njit(cache=True)
def no_bounding(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create a matrix with no bounding.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Bounding matrix where the values inside the bound are finite values (0s) and
        outside the bounds are infinity (non finite).
    """
    return np.zeros((x.shape[1], y.shape[1]))


@njit(cache=True)
def sakoe_chiba(x: np.ndarray, y: np.ndarray, window: float) -> np.ndarray:
    """Create a sakoe chiba lower bounding window on a matrix.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    window: float
        Float that is the size of the window. Must be between 0 and 1.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Sakoe Chiba bounding matrix where the values inside the bound are finite
        values (0s) and outside the bounds are infinity (non finite).

    Raises
    ------
    ValueError
        If the sakoe_chiba_window_radius is not an integer.
    """
    if window < 0 or window > 1:
        raise ValueError("Window must between 0 and 1")

    x_size = x.shape[1]
    y_size = y.shape[1]
    bounding_matrix = np.full((x_size, y_size), np.inf)
    sakoe_chiba_window_radius = ((x_size / 100) * window) * 100

    x_upper_line_values = np.interp(
        list(range(x_size)),
        [0, x_size - 1],
        [0 - sakoe_chiba_window_radius, y_size - sakoe_chiba_window_radius - 1],
    )
    x_lower_line_values = np.interp(
        list(range(x_size)),
        [0, x_size - 1],
        [0 + sakoe_chiba_window_radius, y_size + sakoe_chiba_window_radius - 1],
    )

    bounding_matrix = create_shape_on_matrix(
        bounding_matrix, x_upper_line_values, x_lower_line_values
    )

    return bounding_matrix


@njit(cache=True)
def itakura_parallelogram(
    x: np.ndarray, y: np.ndarray, itakura_max_slope: float
) -> np.ndarray:
    """Create a itakura parallelogram bounding matrix.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    itakura_max_slope: float or int
        Gradient of the slope must be between 0 and 1.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Sakoe Chiba bounding matrix where the values inside the bound are finite
        values (0s) and outside the bounds are infinity (non finite).

    Raises
    ------
    ValueError
        If the itakura_max_slope is not a float or int.
    """
    if itakura_max_slope < 0 or itakura_max_slope > 1:
        raise ValueError("Window must between 0 and 1")
    x_size = x.shape[1]
    y_size = y.shape[1]
    bounding_matrix = np.full((y_size, x_size), np.inf)
    itakura_max_slope = math.floor(((x_size / 100) * itakura_max_slope) * 100) / 2

    middle_x_upper = math.ceil(x_size / 2)
    middle_x_lower = math.floor(x_size / 2)
    if middle_x_lower == middle_x_upper:
        middle_x_lower = middle_x_lower - 1
    middle_y = math.floor(y_size / 2)

    difference_from_middle_y = abs((middle_x_lower * itakura_max_slope) - middle_y)
    middle_y_lower = middle_y + difference_from_middle_y
    middle_y_upper = middle_y - difference_from_middle_y

    x_upper_line_values = np.interp(
        list(range(x_size)),
        [0, middle_x_lower, middle_x_upper, x_size - 1],
        [0, middle_y_upper, middle_y_upper, y_size - 1],
    )
    x_lower_line_values = np.interp(
        list(range(x_size)),
        [0, middle_x_lower, middle_x_upper, x_size - 1],
        [0, middle_y_lower, middle_y_lower, y_size - 1],
    )

    if np.array_equal(x_upper_line_values, x_lower_line_values):
        x_upper_line_values = _check_line_steps(x_upper_line_values)

    bounding_matrix = create_shape_on_matrix(
        bounding_matrix, x_upper_line_values, x_lower_line_values
    )

    return bounding_matrix


@njit(cache=True)
def numba_create_bounding_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float = -1.0,
    itakura_max_slope: float = -1.0,
) -> np.ndarray:
    """Numba compiled way of creating bounding matrix.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    window: float, defaults = -1.
        Float that is the % radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Must be between 0 and 1.
    itakura_max_slope: float, defaults = -1.
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding). Must be between 0 and 1.
    """
    if window != -1.0:
        bounding_matrix = sakoe_chiba(x, y, window)
    elif itakura_max_slope != -1.0:
        bounding_matrix = itakura_parallelogram(x, y, itakura_max_slope)
    else:
        bounding_matrix = no_bounding(x, y)

    return bounding_matrix
