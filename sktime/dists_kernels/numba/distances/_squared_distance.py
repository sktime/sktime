# -*- coding: utf-8 -*-
"""Squared distance and pairwise squared distance."""

__author__ = ["chrisholder"]
__all__ = [
    "squared_distance",
    "pairwise_squared_distance",
    "numba_squared_distance_factory",
]

from typing import Callable

import numpy as np
from numba import njit, prange

from sktime.dists_kernels._utils import to_numba_timeseries
from sktime.dists_kernels.numba.distances.pairwise_distance import pairwise_distance


def squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the squared distance between two timeseries.

    Squared distance is supported for 1d, 2d and 3d arrays.

    The squared distance between two timeseries is defined as:

    .. math::
        sd(x, y) = \sum_{i=1}^{n} (x_i - y_i)^2

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.

    Returns
    -------
    distance: float
        Squared distance between the two timeseries.
    """
    _x = to_numba_timeseries(x)
    _y = to_numba_timeseries(y)
    return _numba_squared_distance(_x, _y)


def pairwise_squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Compute the Squared pairwise distance between two timeseries.

    Pairwise Squared distance is supported for 1d, 2d and 3d arrays.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and m is len(y)
        Pairwise Squared distance matrix of size nxm where n is len(x) and m is
        len(y).
    """
    return pairwise_distance(
        x, y, numba_distance_factory=numba_squared_distance_factory
    )


def numba_squared_distance_factory(
    x: np.ndarray, y: np.ndarray, symmetric: bool, **kwargs: dict
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create a numba compiled squared distance callable based on parameters.

    While in this example parameters aren't used and the already defined numba method
    is returned, in more complex examples to compile them the parameters are needed.
    As such, for consistency parameters are kept here.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        First timeseries.
    y: np.ndarray (1d, 2d or 3d array)
        Second timeseries.
    symmetric: bool
        Boolean that is true when x == y and false when x != y. Used in some instances
        to speed up pairwise computation.
    kwargs: dict
        Extra kwargs. For squared there are none however, this is kept for
        consistency.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], float]
        Numba compiled squared distance callable.
    """
    return _numba_squared_distance


@njit(cache=True, parallel=True)
def _numba_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Numba compiled squared distance.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First timeseries.
    y: np.ndarray (2d array)
        Second timeseries.

    Returns
    -------
    distance: float
        Squared distance between the two timeseries.
    """
    distance = 0.0
    for i in prange(x.shape[0]):
        curr = x[i] - y[i]
        distance += np.sum(curr * curr)

    return distance
