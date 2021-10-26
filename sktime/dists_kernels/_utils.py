# -*- coding: utf-8 -*-
from typing import Callable, Tuple

import numpy as np


def _check_pairwise_timeseries(x: np.ndarray) -> np.ndarray:
    """Check and format a timeseries for pairwise calculation.

    Parameters
    ----------
    x: np.ndarray (1D, 2D or 3D array)
        Timeseries for pairwise.

    Returns
    -------
    validated_x: np.ndarray (3D array)
        Validated timeseries in 3D numpy array format.
    """
    if x.ndim <= 1:
        _x = to_numba_timeseries(x)
    else:
        _x = x

    if _x.ndim <= 2:
        validated_x = np.reshape(_x, _x.shape + (1,))
    else:
        validated_x = _x

    return validated_x


def validate_pairwise_params(
    x: np.ndarray,
    y: np.ndarray = None,
    factory: Callable = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Validate pairwise parameters.

    Parameters
    ----------
    x: np.ndarray (1D, 2D or 3D array)
        First timeseries.
    y: np.ndarray (1D, 2D or 3D array)
        Second timeseries.
    factory: Callable[
        [np.ndarray, np.ndarray, bool, dict],
        Callable[[np.ndarray, np.ndarray], float]
    ]
        Numba factory used to generate numba compiled functions to be used.

    Returns
    -------
    np.ndarray (3D array)
        First validated timeseries. 3D numpy array.
    np.ndarray (3D array)
        Second validated timeseries. 3D numpy array.
    symmetric: bool
        Boolean that is true when x == y and false when x != y. Used in some instances
        to speed up pairwise computation.
    """
    if factory is None:
        raise ValueError("You must specify a numba_distance_factory")

    if y is None:
        y = np.copy(x)
        symmetric = True
    else:
        symmetric = np.array_equal(x, y)

    validated_x = _check_pairwise_timeseries(x)
    validated_y = _check_pairwise_timeseries(y)

    return validated_x, validated_y, symmetric


def to_numba_timeseries(x):
    """Convert a timeseries to a valid timeseries for numba use.

    Parameters
    ----------
    x: np.ndarray (1d or 2d array)
        A timeseries.

    Returns
    -------
    np.ndarray (2d array)
        2d array that is a validated and formatted timeseries for numba use.
    """
    _x = np.array(x, copy=True, dtype=np.float)
    if _x.ndim < 2:
        _x = np.reshape(x, (-1, 1))
    return _x
