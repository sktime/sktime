# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange

from sktime.dists_kernels.numba.distances.base import DistanceCallable


# This seems to be slower maybe need an njit when above a certain num points and just
# python when under certain num points
# @njit(cache=True)
def _compute_distance(
    x: np.ndarray, y: np.ndarray, distance_callable: DistanceCallable
) -> float:
    """Compute distance between two 3d numpy array.

    Parameters
    ----------
    x: np.ndarray (3d array)
        First timeseries.
    y: np.ndarray (3d array)
        Second timeseries.
    distance_callable: Callable[[np.ndarray, np.ndarray], float]
        No_python distance callable to measure the distance between two 2d numpy
        arrays.

    Returns
    -------
    float
        Distance between two timeseries.
    """
    loop_to = min(x.shape[0], y.shape[0])

    total_distance = 0.0

    for i in range(loop_to):
        total_distance += distance_callable(x[i], y[i])

    return total_distance


@njit(cache=True)
def _check_numba_pairwise_series(x: np.ndarray):
    if x.ndim == 2:
        shape = x.shape
        _x = np.reshape(x, (shape[0], shape[1], 1))
    else:
        _x = x
    return _x


@njit(cache=True, parallel=True)
def _compute_pairwise_distance(
    x: np.ndarray, y: np.ndarray, symmetric: bool, distance_callable: DistanceCallable
) -> np.ndarray:
    """Compute pairwise distance between two 3d numpy array.

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

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix[i, j] = pairwise_matrix[j, i]
            else:
                pairwise_matrix[i, j] = distance_callable(curr_x, _y[j])

    return pairwise_matrix
