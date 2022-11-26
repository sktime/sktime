#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Functions for checking input data."""

__author__ = ["mloning", "Drishti Bhasin", "khrapovs"]
__all__ = [
    "check_series",
    "check_time_index",
    "check_equal_time_index",
    "check_consistent_index_type",
]

from typing import Union

import numpy as np
import pandas as pd

# We currently support the following types for input data and time index types.
VALID_DATA_TYPES = (pd.DataFrame, pd.Series, np.ndarray)
VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex, pd.TimedeltaIndex)
RELATIVE_INDEX_TYPES = (pd.RangeIndex, pd.TimedeltaIndex)
ABSOLUTE_INDEX_TYPES = (pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)
assert set(RELATIVE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)
assert set(ABSOLUTE_INDEX_TYPES).issubset(VALID_INDEX_TYPES)


def is_integer_index(x) -> bool:
    """Check that the input is an integer pd.Index."""
    return isinstance(x, pd.Index) and x.is_integer()


def is_in_valid_index_types(x) -> bool:
    """Check that the input type belongs to the valid index types."""
    return isinstance(x, VALID_INDEX_TYPES) or is_integer_index(x)


def is_in_valid_relative_index_types(x) -> bool:
    return isinstance(x, RELATIVE_INDEX_TYPES) or is_integer_index(x)


def is_in_valid_absolute_index_types(x) -> bool:
    return isinstance(x, ABSOLUTE_INDEX_TYPES) or is_integer_index(x)


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
    index: Union[pd.Index, np.array],
    allow_empty: bool = False,
    enforce_index_type: bool = None,
    var_name: str = "input",
) -> pd.Index:
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
    if not is_in_valid_index_types(index):
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )

    if enforce_index_type and type(index) is not enforce_index_type:
        raise NotImplementedError(
            f"{type(index)} is not supported for {var_name}, use "
            f"type: {enforce_index_type} or integer pd.Index instead."
        )

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
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


def check_equal_time_index(*ys, mode="equal"):
    """Check that time series have the same (time) indices.

    Parameters
    ----------
    *ys : tuple of sktime compatible time series data containers
        must be pd.Series, pd.DataFrame or 1/2D np.ndarray, or None
        can be Series, Panel, Hierarchical, but must be pandas or numpy
        note: this assumption is not checked by the function itself
            if check is needed, use check_is_scitype or check_is_mtype before call
    mode : str, "equal" or "contained", optional, default = "equal"
        if "equal" will check for all indices being exactly equal
        if "contained", will check whether all indices are subset of ys[0].index

    Raises
    ------
    ValueError
        if mode = "equal", raised if there are at least two non-None entries of ys
            of which pandas indices are not the same
        if mode = "contained, raised if there is at least one non-None ys[i]
            such that ys[i].index is not contained in ys[o].index
        np.ndarray are considered having (pandas) integer range index on axis 0
    """
    from sktime.datatypes._utilities import get_index_for_series

    # None entries are ignored
    y_not_None = [y for y in ys if y is not None]

    # if there is no or just one element, there is nothing to compare
    if len(y_not_None) < 2:
        return None

    # only validate indices if data is passed as pd.Series
    first_index = get_index_for_series(y_not_None[0])

    for i, y in enumerate(y_not_None[1:]):
        y_index = get_index_for_series(y)

        if mode == "equal":
            failure_cond = not first_index.equals(y_index)
            msg = (
                f"(time) indices are not the same, series 0 and {i} "
                f"differ in the following: {first_index.symmetric_difference(y_index)}."
            )
        elif mode == "contains":
            failure_cond = not y_index.isin(first_index).all()
            msg = (
                f"(time) indices of series {i} are not contained in index of series 0,"
                f" extra indices are: {y_index.difference(first_index)}"
            )
        else:
            raise ValueError('mode must be "equal" or "contains"')

        if failure_cond:
            raise ValueError(msg)


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

    if is_integer_index(a):
        if not is_integer_index(b):
            raise TypeError(msg)

    else:
        # check types, note that isinstance() does not work here because index
        # types inherit from each other, hence we check for type equality
        if not type(a) is type(b):  # noqa
            raise TypeError(msg)
