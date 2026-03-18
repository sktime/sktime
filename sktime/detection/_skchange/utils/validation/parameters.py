"""Common validation functions for input parameters."""

from numbers import Number

import numpy as np
import pandas as pd


def check_none(value: Number, name: str, allow_none: bool = False) -> Number:
    """Check if value is None.

    Parameters
    ----------
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is None and allow_none is False.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    return value


def check_larger_than(
    min_value: Number, value: Number, name: str, allow_none: bool = False
) -> Number:
    """Check if `value` is larger than or equal to `min_value`.

    Parameters
    ----------
    min_value : int, float
        Minimum allowed value.
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is `not None` and smaller than `min_value`.
    """
    check_none(value, name, allow_none)
    if value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value} ({name}={value}).")
    return value


def check_smaller_than(
    max_value: Number, value: Number, name: str, allow_none: bool = False
) -> Number:
    """Check if value is non-negative.

    Parameters
    ----------
    max_value : int, float
        Maximum allowed value.
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is negative.
    """
    check_none(value, name, allow_none)
    if value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value} ({name}={value}).")
    return value


def check_in_interval(
    interval: pd.Interval,
    value: Number,
    name: str,
    allow_none: bool = False,
) -> Number:
    """Check if value is non-negative.

    Parameters
    ----------
    interval : pd.Interval
        Interval to check.
    value : int, float
        Value to check.
    name : str
        Name of the parameter to be shown in the error message.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int, float
        Input value.

    Raises
    ------
    ValueError
        If value is negative.
    """
    check_none(value, name, allow_none)
    if value is not None and value not in interval:
        raise ValueError(f"{name} must be in {interval} ({name}={value}).")
    return value


def check_data_column(
    data_column: int | str,
    column_role: str,
    X: np.ndarray,
    X_columns: pd.Index | None,
) -> int:
    """Check that a data column name or index is valid.

    Parameters
    ----------
    data_column : int or str
        Column index or name to check.
    column_role : str
        Role of the column (e.g., "Response").
    X : np.ndarray
        Data array.
    X_columns : pd.Index or None
        Column names of the data array.

    Returns
    -------
    data_column : int
        Column index.
    """
    if isinstance(data_column, int):
        if not 0 <= data_column < X.shape[1]:
            raise ValueError(
                f"{column_role} column index ({data_column}) must"
                f" be between 0 and {X.shape[1] - 1}."
            )
    elif isinstance(data_column, str) and X_columns is not None:
        if data_column not in X_columns:
            raise ValueError(
                f"{column_role} column ({data_column}) not found "
                f"among the fit data columns: {X_columns}."
            )
        data_column = X_columns.get_loc(data_column)
    else:
        raise ValueError(
            f"{column_role} column must be an integer in the range "
            f"[0, {X.shape[1]}), or a valid column name. Got {data_column}."
        )
    return data_column
