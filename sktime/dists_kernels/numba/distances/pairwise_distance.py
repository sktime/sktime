# -*- coding: utf-8 -*-
"""Pairwise distance between two timeseries."""

__author__ = ["chrisholder"]
__all__ = ["pairwise_distance"]

from typing import Callable, Union

import numpy as np
from numba import njit, prange

from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba.distances.base import DistanceCallable, NumbaDistance
from sktime.dists_kernels.numba.distances.distance import _resolve_metric


def pairwise_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: Union[
        str,
        Callable[
            [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
        ],
        Callable[[np.ndarray, np.ndarray], float],
        NumbaDistance,
    ],
    **kwargs: dict
) -> np.ndarray:
    """Compute the pairwise distance matrix between two timeseries.

    This function works for 1d, 2d and 3d timeseries. No matter the number of dimensions
    passed a 2d array will always be returned.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    metric: str or Callable or NumbaDistance
        The distance metric to use.
        If a string is given, the value must be one of the following strings:

        'euclidean', 'squared', 'dtw.

        If callable then it has to be a distance factory or numba distance callable.
        If the distance takes kwargs then a distance factory should be provided. The
        distance factory takes the form:

        Callable[
            [np.ndarray, np.ndarray, bool, dict],
            Callable[[np.ndarray, np.ndarray], float]
        ]

        and should validate the kwargs, and return a no_python callable described
        above as the return.

        If a no_python callable provided it should take the form:

        Callable[
            [np.ndarray, np.ndarray],
            float
        ],
    kwargs: dict, optional
        Extra arguments for metric. Refer to each metric documentation for a list of
        possible arguments.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix between the two timeseries.

    Raises
    ------
    ValueError
        If the value of x or y provided is not a numpy array.
        If the value of x or y has more than 3 dimensions.
        If a metric string provided, and is not a defined valid string.
        If a metric object (instance of class) is provided and doesn't inherit from
        NumbaDistance.
        If a resolved metric is not no_python compiled.
        If the metric type cannot be determined.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    symmetric = np.array_equal(_x, _y)

    _metric_callable = _resolve_metric(metric, _x, _y, **kwargs)
    return _compute_pairwise_distance(_x, _y, symmetric, _metric_callable)


@njit(cache=True, parallel=True)
def _compute_pairwise_distance(
    x: np.ndarray, y: np.ndarray, symmetric: bool, distance_callable: DistanceCallable
) -> np.ndarray:
    """Compute pairwise distance between two 3d numpy array.

    Parameters
    ----------
    x: np.ndarray (3d array)
        First timeseries.
    y: np.ndarray (3d array)
        Second timeseries.
    symmetric: bool
        Boolean that is true when x == y and false when x != y. Used in some instances
        to speed up pairwise computation.
    distance_callable: Callable[[np.ndarray, np.ndarray], float]
        No_python distance callable to measure the distance between two 2d numpy
        arrays.

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
                pairwise_matrix[i, j] = distance_callable(curr_x, y[j])

    return pairwise_matrix
