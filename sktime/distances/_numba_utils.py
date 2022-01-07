# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
from numba import njit

from sktime.distances.base import DistanceCallable


@njit(cache=True)
def _check_numba_pairwise_series(x: np.ndarray) -> np.ndarray:
    """Check a potential series being passed into pairwise.

    Parameters
    ----------
    x: np.ndarray
        A timeseries

    Returns
    -------
    np.ndarray
        Validated and reshaped (where appropriate) timeseries.
    """
    if x.ndim == 2:
        shape = x.shape
        _x = np.reshape(x, (shape[0], shape[1], 1))
    else:
        _x = x
    return _x


@njit()
def _compute_pairwise_distance(
    x: np.ndarray, y: np.ndarray, symmetric: bool, distance_callable: DistanceCallable
) -> np.ndarray:
    """Compute pairwise distance between two numpy arrays.

    Parameters
    ----------
    x: np.ndarray (2d or 3d array)
        First timeseries.
    y: np.ndarray (2d or 3d array)
        Second timeseries.
    symmetric: bool
        Boolean that is true when x == y and false when x != y. Used in some instances
        to speed up pairwise computation.
    distance_callable: Callable[[np.ndarray, np.ndarray], float]
        No_python distance callable to measure the distance between two 2d numpy
        arrays.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y)).
        Pairwise distance matrix between the two timeseries.
    """
    _x = _check_numba_pairwise_series(x)
    _y = _check_numba_pairwise_series(y)

    x_size = _x.shape[0]
    y_size = _y.shape[0]

    pairwise_matrix = np.zeros((x_size, y_size))

    for i in range(x_size):
        curr_x = _x[i]

        for j in range(y_size):
            if symmetric and j < i:
                pairwise_matrix[i, j] = pairwise_matrix[j, i]
            else:
                pairwise_matrix[i, j] = distance_callable(curr_x, _y[j])

    return pairwise_matrix


def is_no_python_compiled_callable(
    no_python_callable: Callable, raise_error: bool = False
):
    """Check if a callable is no_python compiled.

    Parameters
    ----------
    no_python_callable: Callable
        Callable to check if no_python compiled.
    raise_error: bool, defaults = False
        Boolean that when True will raise an error if the callable is not no_python
        compiled.

    Returns
    -------
    bool
        True if the callable is no_python compiled, False if the callable is not
        no_python compiled

    Raises
    ------
    ValueError
        If the raise_error parameter is True and the callable passed is not no_python
        compiled.
    """
    is_no_python_callable = hasattr(no_python_callable, "signatures")
    if raise_error and not is_no_python_callable:
        raise ValueError(
            f"The callable provided must be no_python compiled. The callable that "
            f"caused"
            f"this error is named {no_python_callable.__name__}"
        )

    return is_no_python_callable


def to_numba_pairwise_timeseries(x: np.ndarray) -> np.ndarray:
    """Convert a timeseries to a valid timeseries for numba pairwise use.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        A timeseries.

    Returns
    -------
    np.ndarray (3d array)
        3d array that is the formatted pairwise timeseries.

    Raises
    ------
    ValueError
        If the value provided is not a numpy array
        If the matrix provided is greater than 3 dimensions
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(
            f"The value {x} is an invalid timeseries. To perform a "
            f"distance computation a numpy arrays must be provided."
        )

    _x = np.array(x, copy=True, dtype=np.float)
    num_dims = _x.ndim
    if num_dims == 1:
        shape = _x.shape
        _x = np.reshape(_x, (1, 1, shape[0]))
    elif num_dims == 2:
        shape = _x.shape
        _x = np.reshape(_x, (shape[0], shape[1], 1))
    elif num_dims == 3:
        shape = _x.shape
        _x = np.reshape(_x, (shape[0], shape[1], shape[2]))
    elif num_dims > 3:
        raise ValueError(
            "The matrix provided has more than 3 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "3 dimensions"
        )
    return _x


def to_numba_timeseries(x: np.ndarray) -> np.ndarray:
    """Convert a timeseries to a valid timeseries for numba use.

    Parameters
    ----------
    x: np.ndarray (1d or 2d)
        A timeseries.

    Returns
    -------
    np.ndarray (2d array)
        2d array that is the formatted timeseries.

    Raises
    ------
    ValueError
        If the value provided is not a numpy array
        If the matrix provided is greater than 2 dimensions
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(
            f"The value {x} is an invalid timeseries. To perform a "
            f"distance computation a numpy arrays must be provided."
        )

    _x = np.array(x, copy=True, dtype=np.float)
    num_dims = _x.ndim
    shape = _x.shape
    if num_dims == 1:
        _x = np.reshape(_x, (shape[0], 1))
    if num_dims == 2 and shape[0] == 1:
        _x = np.reshape(_x, (shape[1], shape[0]))

    elif num_dims > 2:
        raise ValueError(
            "The matrix provided has more than 2 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "2 dimensions"
        )
    return _x
