# -*- coding: utf-8 -*-
from typing import Callable, Tuple
import numpy as np
from numba import njit, prange
from sktime.dists_kernels.numba_distances.squared_distance import (
    _numba_squared_distance,
)


def check_pairwise_params(
    x: np.ndarray, y: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Method used to check the params of x and y to ensure readiness for pairwise.

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List, defaults = None
        Second matrix of multiple time series.

    Returns
    -------
    validated_x: np.ndarray
        First validated time series
    validated_y: np.ndarray
        Second validated time series
    symmetric: bool
        Boolean marking if the pairwise will be symmetric (if the two timeseries are
        equal)
    """
    if y.size is None:
        y = np.copy(x)
        symmetric = True
    else:
        symmetric = np.array_equal(x, y)

    if x.ndim <= 2:
        validated_x = np.reshape(x, x.shape + (1,))
        validated_y = np.reshape(y, y.shape + (1,))
    else:
        validated_x = x
        validated_y = y
    return validated_x, validated_y, symmetric


def pairwise(
    x: np.ndarray,
    y: np.ndarray = None,
    numba_distance_factory: Callable = _numba_squared_distance,
    **kwargs: dict
) -> np.ndarray:
    """
    Method to compute a pairwise distance on a matrix (i.e. distance between each
    ts in the matrix)
    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List, defaults = None
        Second matrix of multiple time series.
    numba_distance_factory: Callable
        Factory to create a numba callable using kwargs
    **kwargs: dict
        kwargs for the pairwise function. See arguments for distance you're using
        for valid kwargs
    Returns
    -------
    np.ndarray
        Matrix containing the pairwise distance between each point
    """
    validated_x, validated_y, symmetric = check_pairwise_params(x, y)
    kwargs["symmetric"] = symmetric
    distance: Callable[[np.ndarray, np.ndarray], float] = numba_distance_factory(
        x, y, **kwargs
    )

    return _numba_pairwise(validated_x, validated_y, symmetric, distance)


@njit(parallel=True)
def _numba_pairwise(
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
        Matrix containing the pairwise distance between the two matrices
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
