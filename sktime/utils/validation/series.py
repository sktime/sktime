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
import numpy as np
import pandas as pd

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


def _check_is_univariate(y, var_name="input"):
    """Check if series is univariate."""
    if isinstance(y, pd.DataFrame):
        nvars = y.shape[1]
        if nvars > 1:
            raise ValueError(
                f"{var_name} must be univariate, but found {nvars} variables."
            )
    if isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(
            f"{var_name} must be univariate, but found np.ndarray with more than "
            "one column"
        )


def _check_is_multivariate(Z, var_name="input"):
    """Check if series is multivariate."""
    if isinstance(Z, pd.Series):
        raise ValueError(f"{var_name} must have 2 or more variables, but found 1.")
    if isinstance(Z, pd.DataFrame):
        nvars = Z.shape[1]
        if nvars < 2:
            raise ValueError(
                f"{var_name} must have 2 or more variables, but found {nvars}."
            )
    if isinstance(Z, np.ndarray):
        if Z.ndim == 1 or (Z.ndim == 2 and Z.shape[1] == 1):
            raise ValueError(f"{var_name} must have 2 or more variables, but found 1.")


def check_series(
    Z,
    enforce_univariate=False,
    enforce_multivariate=False,
    allow_empty=False,
    allow_numpy=True,
    allow_None=True,
    enforce_index_type=None,
    allow_index_names=False,
    var_name="input",
):
    """Validate input data to be a valid mtype for Series.

    Parameters
    ----------
    Z : pd.Series, pd.DataFrame, np.ndarray, or None
        Univariate or multivariate time series.
    enforce_univariate : bool, default = False
        If True, multivariate Z will raise an error.
    enforce_multivariate: bool, default = False
        If True, univariate Z will raise an error.
    allow_empty : bool, default = False
        whether a container with zero samples is allowed
    allow_numpy : bool, default = True
        whether no error is raised if Z is in a valid numpy.ndarray format
    allow_None : bool, default = True
        whether no error is raised if Z is None
    enforce_index_type : type, default = None
        type of time index
    allow_index_names : bool, default = False
        If False, names of Z.index will be set to None
    var_name : str, default = "input" - variable name printed in error messages

    Returns
    -------
    Z : pd.Series, pd.DataFrame, np.ndarray, or None
        Validated time series - a reference to the input Z

    Raises
    ------
    TypeError - if Z is not in a valid type or format for scitype Series
    if enforce_univariate is True:
        ValueError if Z has 2 or more columns
    if enforce_multivariate is True:
        ValueError if Z has 1 column
    if allow_numpy is false:
        TypeError - if Z is of type np.ndarray
    if allow_empty is false:
        ValueError - if Z has length 0
    if allow_None is false:
        ValueError - if Z is None
    if enforce_index_type is not None and Z is pandas type:
        ValueError - if Z has index type other than enforce_index_type
    """
    if Z is None:
        if allow_None:
            return Z
        else:
            raise ValueError(var_name + " cannot be None")

    # Check if pandas series or numpy array
    if not allow_numpy:
        valid_data_types = tuple(
            filter(lambda x: x is not np.ndarray, VALID_DATA_TYPES)
        )
    else:
        valid_data_types = VALID_DATA_TYPES

    if not isinstance(Z, valid_data_types):
        raise TypeError(
            f"{var_name} must be a one of {valid_data_types}, but found type: {type(Z)}"
        )

    if enforce_univariate and enforce_multivariate:
        raise ValueError(
            "`enforce_univariate` and `enforce_multivariate` cannot both be set to "
            "True."
        )

    if enforce_univariate:
        _check_is_univariate(Z, var_name=var_name)

    if enforce_multivariate:
        _check_is_multivariate(Z, var_name=var_name)

    # check time index if input data is not an NumPy ndarray
    if not isinstance(Z, np.ndarray):
        check_time_index(
            Z.index,
            allow_empty=allow_empty,
            enforce_index_type=enforce_index_type,
            var_name=var_name,
        )

    if not allow_index_names and not isinstance(Z, np.ndarray):
        Z.index.names = [None for name in Z.index.names]

    return Z


def check_time_index(
    index, allow_empty=False, enforce_index_type=None, var_name="input"
):
    """Check time index.

    Parameters
    ----------
    index : pd.Index or np.array
        Time index
    allow_empty : bool, optional (default=False)
        If False, empty `index` raises an error.
    enforce_index_type : type, optional (default=None)
        type of time index
    var_name : str, default = "input" - variable name printed in error messages

    Returns
    -------
    time_index : pd.Index
        Validated time index - a reference to the input index
    """
    if isinstance(index, np.ndarray):
        index = pd.Index(index)

    # We here check for type equality because isinstance does not
    # work reliably because index types inherit from each other.
    if not type(index) in VALID_INDEX_TYPES:
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )

    if enforce_index_type and type(index) is not enforce_index_type:
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"type: {enforce_index_type} instead."
        )

    # Check time index is ordered in time
    if not index.is_monotonic:
        raise ValueError(
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )

    # Check that index is not empty
    if not allow_empty and len(index) < 1:
        raise ValueError(
            f"{var_name} must contain at least some values, but found none."
        )

    return index


def check_equal_time_index(*ys):
    """Check that time series have the same (time) indices.

    Parameters
    ----------
    *ys : tuple of pd.Series, pd.DataFrame or np.ndarray, or None
        One or more time series

    Raises
    ------
    ValueError
        If there are at least two no=-None entries of ys
            of which pandas indices are not the same
            np.ndarray are considered having integer range index on axis 0
    """
    # None entries are ignored
    y_not_None = [y for y in ys if y is not None]

    # if there is no or just one element, there is nothing to compare
    if len(y_not_None) < 2:
        return None

    # only validate indices if data is passed as pd.Series
    if isinstance(y_not_None[0], np.ndarray):
        first_index = pd.Index(range(len(y_not_None[0])))
    else:
        first_index = y_not_None[0].index

    check_time_index(first_index)

    for y in y_not_None[1:]:
        if isinstance(y, np.ndarray):
            y_index = pd.Index(y)
        else:
            y_index = y.index

        check_time_index(y_index)

        if not first_index.equals(y_index):
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
