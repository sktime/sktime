#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = ["check_series"]

import numpy as np
import pandas as pd

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


def check_series(y, enforce_univariate=False, allow_empty=False):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series, pd.DataFrame
        Univariate or multivariate time series
    enforce_univariate : bool, optional (default=False)
        If True, multivariate Z will raise an error.
    allow_empty : bool

    Returns
    -------
    y : pd.Series, pd.DataFrame
        Validated time series

    Raises
    ------
    ValueError, TypeError
        If Z is an invalid input
    """
    # Check if pandas series or numpy array
    if enforce_univariate and not isinstance(y, pd.Series):
        raise ValueError("Data must be univariate, but found a pd.DataFrame")

    if not isinstance(y, VALID_DATA_TYPES):
        raise TypeError(
            f"Data must be a pandas Series or DataFrame, but found type: {type(y)}"
        )

    # check time index
    check_time_index(y.index, allow_empty=allow_empty)
    return y


def check_time_index(index, allow_empty=False):
    """Check time index.

    Parameters
    ----------
    index : pd.Index or np.array
        Time index
    allow_empty : bool, optional (default=False)
        If True, empty `index` raises an error.

    Returns
    -------
    time_index : pd.Index
        Validated time index
    """
    if isinstance(index, np.ndarray):
        index = pd.Index(index)

    # We here check for type equality because isinstance does not work reliably
    # because index types inherit from each other.
    if not type(index) in VALID_INDEX_TYPES:
        raise NotImplementedError(
            f"{type(index)} is not supported, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )

    # Check time index is ordered in time
    if not index.is_monotonic:
        raise ValueError(
            f"The (time) index must be sorted (monotonically increasing), "
            f"but found: {index}"
        )

    # Check that index is not empty
    if not allow_empty and len(index) < 1:
        raise ValueError(
            f"`index` must contain at least some values, but found "
            f"empty index: {index}."
        )

    return index


def check_equal_time_index(*ys):
    """Check that time series have the same (time) indices.

    Parameters
    ----------
    ys : pd.Series or pd.DataFrame
        One or more time series

    Raises
    ------
    ValueError
        If (time) indices are not the same
    """

    # only validate indices if data is passed as pd.Series
    first_index = ys[0].index
    check_time_index(first_index)

    for y in ys[1:]:
        check_time_index(y.index)

        if not first_index.equals(y.index):
            raise ValueError("Some (time) indices are not the same.")
