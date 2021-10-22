# -*- coding: utf-8 -*-
from typing import Callable, Any, Tuple
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
    """
    Method that takes a distance function and computes the pairwise distance
    Parameters
    ----------
    x: np.ndarray
        First 3d numpy array containing multiple timeseries
    y: np.ndarray
        Second 3d numpy array containing multiple timeseries
    symmetric: bool
        Boolean when true means the two matrices are the same
    distance: Callable[[np.ndarray, np.ndarray], float]
        Callable that is a distance function to measure the distance between two
        time series. NOTE: This must be a numba compatible function (i.e. @njit)
    Returns
    -------
    np.ndarray
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        pairwise distances between each element.
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
        [np.ndarray, np.ndarray, Any], Callable[[np.ndarray, np.ndarray], float]
    ] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    Method to compute a pairwise distance between two timeseries or two timeseries
    panels.
    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List, defaults = None
        Second matrix of multiple time series.
    param_validator: Callable[
        [np.ndarray, np.ndarray, dict],
        Tuple[np.ndarray, np.ndarray, dict]
    ], defaults = None
        Method used to validate and return the validated parameters
    numba_distance_factory: Callable, defaults = None
        Factory to create a numba callable that takes (x, y, **kwargs) using kwargs
    **kwargs: Any
        kwargs for the pairwise function. See arguments for distance you're using
        for valid kwargs

    Returns
    -------
    np.ndarray
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        pairwise distances between each element.
    """
    if param_validator is not None:
        x, y, kwargs = param_validator(x, y, **kwargs)

    validated_x, validated_y, symmetric = validate_pairwise_params(
        x, y, numba_distance_factory
    )
    kwargs_dict = {"symmetric": symmetric, **kwargs}

    distance_func: Callable[[np.ndarray, np.ndarray], float] = numba_distance_factory(
        validated_x, validated_y, **kwargs_dict
    )

    return _numba_pairwise_distance(validated_x, validated_y, symmetric, distance_func)
