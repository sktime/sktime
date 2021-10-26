# -*- coding: utf-8 -*-
"""Pairwise distance between two timeseries."""

__author__ = ["chrisholder"]
__all__ = ["pairwise_distance"]

from typing import Any, Callable, Tuple

import numpy as np
from numba import njit, prange

from sktime.dists_kernels._utils import validate_pairwise_params


@njit(parallel=True)
def _numba_pairwise_distance(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    distance: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Numba compiled pairwise distance.

    Parameters
    ----------
    x: np.ndarray (3d array)
        First timeseries.
    y: np.ndarray (3d array)
        Second timeseries.
    symmetric: bool
        Boolean that is true when x == y and false when x != y. Used in some instances
        to speed up pairwise computation.
    distance: Callable[[np.ndarray, np.ndarray], float]
        Numba compiled distance that accepts numpy 2d array as first parameter and
        numpy 2d array as second parameters. The function will returns a float.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix between x and y. This is of size [n, m] where n
        is len(x) and m is len(y).
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    pairwise_matrix = np.zeros((x_size, y_size))

    for i in range(x_size):
        curr_x = x[i]

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix[i, j] = pairwise_matrix[j, i]
            else:
                pairwise_matrix[i, j] = distance(curr_x, y[j])

    return pairwise_matrix


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray = None,
    param_validator: Callable[
        [np.ndarray, np.ndarray, dict], Tuple[np.ndarray, np.ndarray, dict]
    ] = None,
    numba_distance_factory: Callable[
        [np.ndarray, np.ndarray, bool, dict], Callable[[np.ndarray, np.ndarray], float]
    ] = None,
    **kwargs: Any
) -> np.ndarray:
    """Compute a pairwise distance matrix between two timeseries.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    param_validator: Callable[
        [np.ndarray, np.ndarray, dict],
        Tuple[np.ndarray, np.ndarray, dict]
    ], defaults = None
        Method to validate custom parameters.
    numba_distance_factory: Callable, defaults = None
        Method to create a numba callable that takes (x, y, **kwargs) using kwargs.
    **kwargs: Any
        kwargs for the pairwise function. See the distance function parameters
        for valid kwargs.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix between x and y. This is of size [n, m] where n
        is len(x) and m is len(y).
    """
    if param_validator is not None:
        x, y, kwargs = param_validator(x, y, **kwargs)

    validated_x, validated_y, symmetric = validate_pairwise_params(
        x, y, numba_distance_factory
    )

    distance_func: Callable[[np.ndarray, np.ndarray], float] = numba_distance_factory(
        validated_x, validated_y, symmetric, **kwargs
    )

    return _numba_pairwise_distance(validated_x, validated_y, symmetric, distance_func)
