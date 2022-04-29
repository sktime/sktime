# -*- coding: utf-8 -*-
"""Utilities for validating panel data."""

__author__ = ["mloning"]
__all__ = [
    "check_X",
    "check_y",
    "check_X_y",
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length

from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_nested_to_3d_numpy,
)

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
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim == 1 or X.ndim > 3:
            raise ValueError(
                f"If passed as a np.array, X must be a 2 or 3-dimensional "
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
