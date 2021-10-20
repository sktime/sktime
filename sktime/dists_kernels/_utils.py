# -*- coding: utf-8 -*-
from typing import Callable, Tuple
import numpy as np


def _check_pairwise_timeseries(
    x: np.ndarray, y: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Method used to check the params of x and y.

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        First matrix of multiple timeseries
    y: np.ndarray or pd.Dataframe or List, defaults = None
        Second matrix of multiple timeseries.

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


def validate_pairwise_params(
    x: np.ndarray,
    y: np.ndarray = None,
    factory: Callable = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Method used to validate pairwise parameters passed.

    Parameters
    ----------
    x: np.ndarray
        First matrix of multiple timeseries
    y: np.ndarray
        Second matrix of multiple timeseries
    factory: Callable
        Numba factory used to generate numba compiled functions to be used

    Returns
    -------
    np.ndarray
        First validated timeseries
    np.ndarray
        Second validated timeseries
    factory: Callable
        Validated numba factory
    """
    if factory is None:
        raise ValueError("You must specify a numba_distance_factory")

    validated_x, validated_y, symmetric = _check_pairwise_timeseries(x, y)
    return validated_x, validated_y, symmetric


def to_numba_timeseries(x):
    """Method to convert timeseries to a valid timeseries for numba use.

    Parameters
    ----------
    x: np.ndarray
        Any valid panel or series timeseries

    Returns
    -------
    _x: np.ndarray
        Numpy array that has been converted and formatted
    """
    _x = np.array(x, copy=True, dtype=np.float)
    if _x.ndim < 2:
        _x = np.reshape(x, (-1, 1))
    return _x
