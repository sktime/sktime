# -*- coding: utf-8 -*-
from typing import Callable, Any
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


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray = None,
    numba_distance_factory: Callable[
        [np.ndarray, np.ndarray, Any], Callable[[np.ndarray, np.ndarray], float]
    ] = None,
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
    numba_distance_factory: Callable, defaults = None
        Factory to create a numba callable using kwargs
    **kwargs: dict
        kwargs for the pairwise function. See arguments for distance you're using
        for valid kwargs
    Returns
    -------
    np.ndarray
        Matrix containing the pairwise distance between each point
    """
    validated_x, validated_y, symmetric = validate_pairwise_params(
        x, y, numba_distance_factory
    )
    kwargs["symmetric"] = symmetric
    distance: Callable[[np.ndarray, np.ndarray], float] = numba_distance_factory(
        x, y, **kwargs
    )

    return _numba_pairwise_distance(validated_x, validated_y, symmetric, distance)
