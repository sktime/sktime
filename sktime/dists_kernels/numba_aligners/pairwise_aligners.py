# -*- coding: utf-8 -*-
from typing import Tuple, Callable, Any
import numpy as np
from numba import njit, prange

from sktime.dists_kernels._utils import validate_pairwise_params


@njit(parallel=True)
def _numba_alignment_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    dist_func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise for alignment  which processes alignment matrix and distance.

    Parameters
    ----------
    x: np.ndarray
        First 3d numpy array containing multiple timeseries
    y: np.ndarray
        Second 3d numpy array containing multiple timeseries
    symmetric: bool
        Boolean when true means the two matrices are the same
    dist_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]
        Callable that is a alignment function to measure the distance between two
        time series that returns the alignment is used to compute this distance.
        NOTE: This must be a numba compatible function (i.e. @njit)
    Returns
    -------
    np.ndarray
        Matrix containing the cost for each of the pairwise
    np.ndarray
        Matrix containing the pairwise distance between the two matrices
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    x_dims = x.shape[1]
    y_dims = y.shape[1]

    pairwise_matrix_dist = np.zeros((x_size, y_size))
    pairwise_cost_matrix = np.zeros((x_size, y_size, x_dims, y_dims))

    for i in range(x_size):
        curr_x = x[i]

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix_dist[i, j] = pairwise_matrix_dist[j, i]
                pairwise_cost_matrix[i, j] = pairwise_cost_matrix[j, i, :]
            else:
                pairwise_matrix_dist[i, j], pairwise_cost_matrix[i, j] = dist_func(
                    curr_x, y[j]
                )
    return pairwise_cost_matrix, pairwise_matrix_dist


def pairwise_alignment(
    x: np.ndarray,
    y: np.ndarray = None,
    numba_aligner_factory: Callable[
        [np.ndarray, np.ndarray, Any], Callable[[np.ndarray, np.ndarray], float]
    ] = None,
    **kwargs: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method to compute a pairwise alignment between two timeseries or two timeseries
    panels.

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple time series
    y: np.ndarray or pd.Dataframe or List, defaults = None
        Second matrix of multiple time series.
    numba_aligner_factory: Callable, defaults = None
        Factory to create a numba callable that takes (x, y, **kwargs) using kwargs
    **kwargs: dict
        kwargs for the pairwise function. See arguments for distance you're using
        for valid kwargs
    Returns
    -------
    np.ndarray
        [n, m, n_dims, m_dims] where n is the size of x, m is the size of y, n_dims
        is the x dimensions, m_dims is y dimensions; matrix that contains the given
        alignment calculation for each pairwise.
    np.ndarray
        [n, m] where n is the size of x, m is the size of y; matrix that contains the
        pairwise distances between each element.
    """
    validated_x, validated_y, symmetric = validate_pairwise_params(
        x, y, numba_aligner_factory
    )
    kwargs["symmetric"] = symmetric
    alignment_func: Callable[[np.ndarray, np.ndarray], float] = numba_aligner_factory(
        x, y, **kwargs
    )

    return numba_aligner_factory(validated_x, validated_y, symmetric, alignment_func)
