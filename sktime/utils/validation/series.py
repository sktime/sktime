#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Functions for checking input data."""

__author__ = ["Markus Löning", "Drishti Bhasin"]
__all__ = [
    "check_series",
    "check_time_index",
    "check_equal_time_index",
    "check_consistent_index_type",
]
import pandas as pd
import numpy as np

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


def _check_is_univariate(y):
    """Check if series is univariate."""
    if isinstance(y, pd.DataFrame):
        raise ValueError("Data must be univariate, but found a pd.DataFrame")
    if isinstance(y, np.ndarray) and y.ndim > 1:
        raise ValueError(
            "Data must be univariate, but found np.array with more than "
            "one dimension"
        )


def _check_is_multivariate(Z):
    """Check if series is multivariate"""
    if isinstance(Z, pd.Series):
        raise ValueError("Data must be multivariate, but found a pd.Series")
    if isinstance(Z, np.ndarray) and Z.ndim == 1:
        raise ValueError(
            "Data must be multivariate, but found np.array with one dimension"
        )


def check_series(
    Z,
    enforce_univariate=False,
    enforce_multivariate=False,
    allow_empty=False,
    allow_numpy=True,
    enforce_index_type=None,
):
    """Validate input data.

    Parameters
    ----------
    Z : pd.Series, pd.DataFrame, np.ndarray
        Univariate or multivariate time series. NumPy ndarray only allowed
        if `allow_numpy` = True.
    enforce_univariate : bool, default = False
        If True, multivariate Z will raise an error.
    enforce_multivariate: bool, default = False
        If True, univariate Z will raise an error.
    allow_empty : bool, default = False
    allow_numpy : bool, default = True
    enforce_index_type : type, default = None
        type of time index

    Returns
    -------
    Z : pd.Series, pd.DataFrame, np.ndarray
        Validated time series

    Raises
    ------
    ValueError, TypeError
        If Z is an invalid input
    """
    # Check if pandas series or numpy array
    if not allow_numpy:
        valid_data_types = tuple(
            filter(lambda x: x is not np.ndarray, VALID_DATA_TYPES)
        )
    else:
        valid_data_types = VALID_DATA_TYPES

    if not isinstance(Z, valid_data_types):
        raise TypeError(
            f"Data must be a one of {valid_data_types}, but found type: {type(Z)}"
        )

    if enforce_univariate:
        _check_is_univariate(Z)

    if enforce_multivariate:
        _check_is_multivariate(Z)

    # check time index if input data is not an NumPy ndarray
    if not isinstance(Z, np.ndarray):
        check_time_index(
            Z.index, allow_empty=allow_empty, enforce_index_type=enforce_index_type
        )

    return Z


def check_time_index(index, allow_empty=False, enforce_index_type=None):
    """Check time index.

    Parameters
    ----------
    index : pd.Index or np.array
        Time index
    allow_empty : bool, optional (default=False)
        If False, empty `index` raises an error.
    enforce_index_type : type, optional (default=None)
        type of time index

    Returns
    -------
    time_index : pd.Index
        Validated time index
    """
    if isinstance(index, np.ndarray):
        index = pd.Index(index)

    # We here check for type equality because isinstance does not
    # work reliably because index types inherit from each other.
    if not type(index) in VALID_INDEX_TYPES:
        raise NotImplementedError(
            f"{type(index)} is not supported, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )

    if enforce_index_type and type(index) is not enforce_index_type:
        raise NotImplementedError(
            f"{type(index)} is not supported. Please use "
            f"type: {enforce_index_type} instead."
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
    ys : pd.Series, pd.DataFrame or np.ndarray
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


def _is_int_index(index):
    """Check if index type is one of pd.RangeIndex or pd.Int64Index."""
    return type(index) in (pd.Int64Index, pd.RangeIndex)


def check_consistent_index_type(a, b):
    """Check that two indices have consistent types.

    Parameters
    ----------
    a : pd.Index
        Index being checked for consistency
    b : pd.Index
        Index being checked for consistency

    Raises
    ------
    TypeError
        If index types are inconsistent
    """
    msg = (
        "Found series with inconsistent index types, please make sure all "
        "series have the same index type."
    )

    if _is_int_index(a):
        if not _is_int_index(b):
            raise TypeError(msg)

    else:
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        if not type(a) is type(b):  # noqa
            raise TypeError(msg)
