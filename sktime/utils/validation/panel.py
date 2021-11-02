# -*- coding: utf-8 -*-
"""Utilities for validating panel data."""

__author__ = ["mloning"]
__all__ = [
    "check_X",
    "check_y",
    "check_X_y",
    "check_classifier_input",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length

from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.datatypes._panel._check import is_nested_dataframe

VALID_X_TYPES = (pd.DataFrame, np.ndarray)  # nested pd.DataFrame, 2d or 3d np.array
VALID_Y_TYPES = (pd.Series, np.ndarray)  # 1-d vector


def check_X(
    X,
    enforce_univariate=False,
    enforce_min_instances=1,
    enforce_min_columns=1,
    coerce_to_numpy=False,
    coerce_to_pandas=False,
):
    """Validate input data.

    Parameters
    ----------
    X : pd.DataFrame or np.array
        Input data
    enforce_univariate : bool, optional (default=False)
        Enforce that X is univariate.
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.
    enforce_min_columns : int, optional (default=1)
        Enforce minimum number of columns (or time-series variables).
    coerce_to_numpy : bool, optional (default=False)
        If True, X will be coerced to a 3-dimensional numpy array.
    coerce_to_pandas : bool, optional (default=False)
        If True, X will be coerced to a nested pandas DataFrame.

    Returns
    -------
    X : pd.DataFrame or np.array
        Checked and possibly converted input data

    Raises
    ------
    ValueError
        If X is invalid input data
    """
    # check input type
    if coerce_to_pandas and coerce_to_numpy:
        raise ValueError(
            "`coerce_to_pandas` and `coerce_to_numpy` cannot both be set to True"
        )

    if not isinstance(X, VALID_X_TYPES):
        raise ValueError(
            f"X must be a pd.DataFrame or a np.array, " f"but found: {type(X)}"
        )

    # check np.array
    # check first if we have the right number of dimensions, otherwise we
    # may not be able to get the shape of the second dimension below
    if isinstance(X, np.ndarray):
        if not X.ndim == 3:
            raise ValueError(
                f"If passed as a np.array, X must be a 3-dimensional "
                f"array, but found shape: {X.shape}"
            )
        if coerce_to_pandas:
            X = from_3d_numpy_to_nested(X)

    # enforce minimum number of columns
    n_columns = X.shape[1]
    if n_columns < enforce_min_columns:
        raise ValueError(
            f"X must contain at least: {enforce_min_columns} columns, "
            f"but found only: {n_columns}."
        )

    # enforce univariate data
    if enforce_univariate and n_columns > 1:
        raise ValueError(
            f"X must be univariate with X.shape[1] == 1, but found: "
            f"X.shape[1] == {n_columns}."
        )

    # enforce minimum number of instances
    if enforce_min_instances > 0:
        _enforce_min_instances(X, min_instances=enforce_min_instances)

    # check pd.DataFrame
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
        # convert pd.DataFrame
        if coerce_to_numpy:
            X = from_nested_to_3d_numpy(X)

    return X


def check_y(y, enforce_min_instances=1, coerce_to_numpy=False):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.
    coerce_to_numpy : bool, optional (default=False)
        If True, y will be coerced to a numpy array.

    Returns
    -------
    y : pd.Series or np.array
    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if not isinstance(y, VALID_Y_TYPES):
        raise ValueError(
            f"y must be either a pd.Series or a np.ndarray, "
            f"but found type: {type(y)}"
        )

    if enforce_min_instances > 0:
        _enforce_min_instances(y, min_instances=enforce_min_instances)

    if coerce_to_numpy and isinstance(y, pd.Series):
        y = y.to_numpy()

    return y


def check_X_y(
    X,
    y,
    enforce_univariate=False,
    enforce_min_instances=1,
    enforce_min_columns=1,
    coerce_to_numpy=False,
    coerce_to_pandas=False,
):
    """Validate input data.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series or np.array
    enforce_univariate : bool, optional (default=False)
        Enforce that X is univariate.
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.
    enforce_min_columns : int, optional (default=1)
        Enforce minimum number of columns (or time-series variables).
    coerce_to_numpy : bool, optional (default=False)
        If True, X will be coerced to a 3-dimensional numpy array.
    coerce_to_pandas : bool, optional (default=False)
        If True, X will be coerced to a nested pandas DataFrame.

    Returns
    -------
    X : pd.DataFrame or np.array
    y : pd.Series
    Raises
    ------
    ValueError
        If y or X is invalid input data
    """
    # Since we check for consistent lengths, it's enough to
    # only check y for the minimum number of instances
    y = check_y(y, coerce_to_numpy=coerce_to_numpy)
    check_consistent_length(X, y)

    X = check_X(
        X,
        enforce_univariate=enforce_univariate,
        enforce_min_columns=enforce_min_columns,
        enforce_min_instances=enforce_min_instances,
        coerce_to_numpy=coerce_to_numpy,
        coerce_to_pandas=coerce_to_pandas,
    )
    return X, y


def _enforce_min_instances(x, min_instances=1):
    n_instances = x.shape[0]
    if n_instances < min_instances:
        raise ValueError(
            f"Found array with: {n_instances} instance(s) "
            f"but a minimum of: {min_instances} is required."
        )


def check_classifier_input(
        X,
        y,
        enforce_min_instances=1,
        enforce_min_columns=1,
):
    """Check wether input X and y are valid formats with minium data.

    Parameters
    ----------
    X : pd.DataFrame or np.array
    y : pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.
    enforce_min_columns : int, optional (default=1)
        Enforce minimum number of columns (or time-series variables).

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or not enough data
    """
    if not isinstance(y, np.array):
        raise ValueError(
            f"y must be a np.array, "
            f"but found type: {type(y)}"
        )
    if not isinstance(X, pd.pandas):
        if not isinstance(X,np.array):
            raise ValueError(
                f"x must be either a pd.Series or a np.ndarray, "
                f"but found type: {type(x)}"
            )
        else:
            n_cases,  n_dims = X.shape
            if not (n_dims is 2 or n_dims is 3):
                raise ValueError(
                    f"x is an np.array but it must be 2 or 3 dimensional"
                    f"but found to be: {n_dims}"
                )
        if n_cases != y.shape[0]:
            raise ValueError(
                f"unequal number of cases in X and y"
                f"X has : {n_dims}, y has {y.shape[0]}"
            )


def check_classifier_input(
        X,
        y,
        enforce_min_instances=1,
        enforce_min_series_length=1,
):
    """Check wether input X and y are valid formats with minimum data. Raises a
    ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether a pd.DataFrame or np.ndarray
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.
    enforce_min_series_length : int, optional (default=1)
        Enforce minimum series length for input ndarray (i.e. fixed length problems)

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or not enough data
    """
    # Check y
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(
            f"y must be a np.array or a pd.Series, "
            f"but found type: {type(y)}"
        )
    # Check X
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError(
            f"X must be either a pd.DataFrame or a np.ndarray, "
            f"but found type: {type(X)}"
        )
    # Check size of X and y match and minimum data input
    n_cases = X.shape[0]
    n_labels = y.shape[0]
    if isinstance(X, np.ndarray):
        if not (X.ndim is 2 or X.ndim is 3):
            raise ValueError(
                f"x is an np.ndarray, which means it must be 2 or 3 dimensional"
                f"but found to be: {n_dims}"
            )
        if X.ndim is 2 and X.shape[1] < enforce_min_series_length:
            raise ValueError(
                f"x is a 2D np.ndarray, equal length series are length {n_dims}"
                f"but the minimum is  {enforce_min_series_length}"
            )
        if X.ndim is 3 and X.shape[2] < enforce_min_series_length:
            raise ValueError(
                f"x is a 2D np.ndarray, equal length series are length {n_dims}"
                f"but the minimum is  {enforce_min_series_length}"
            )

    else:
        if X.shape[1] is 0:
            raise ValueError(
                f"x is an pd.pandas with no data (num columns == 0)."
            )
    if n_cases < enforce_min_instances:
        raise ValueError(
            f"Minimum number of cases required is {enforce_min_instances} but X "
            f"has : {n_cases}"
        )
    if n_cases != n_labels:
        raise ValueError(
            f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
            f"{n_labels}"
        )


def has_nans(X) -> bool:
    """Check whether an input numpy array has nans."""
    return False
