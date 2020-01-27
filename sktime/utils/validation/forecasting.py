__all__ = ["validate_y", "validate_X", "validate_y_X", "validate_fh", "validate_cv"]
__author__ = "Markus Löning"

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def validate_y_X(y, X):
    """Validate input data.

    Parameters
    ----------
    y : pandas Series or numpy ndarray
    X : pandas DataFrame

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    validate_y(y)
    validate_X(X)


def validate_y(y):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series

    Returns
    -------
    y : pd.Series

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    # Check if pandas series

    if not isinstance(y, pd.Series):
        raise ValueError(f'y must be a pd.Series, but found: {type(y)}')

    # check time index
    validate_time_index(y.index)
    return y


def validate_cv(cv):
    if not hasattr(cv, "split"):
        raise ValueError("Expected cv as a temporal cross-validation object with `split` method")

    if not hasattr(cv, "fh"):
        raise ValueError("Expected cv as a temporal cross-validation object with `fh` attribute")

    return cv


def validate_time_index(time_index):
    """Validate time index

    Parameters
    ----------
    time_index : pd.Index

    Returns
    -------
    time_index : pd.Index
    """
    # period or datetime index are not support yet
    # TODO add support for period/datetime indexing
    if isinstance(time_index, (pd.PeriodIndex, pd.DatetimeIndex)):
        raise NotImplementedError(f"{type(time_index)} is not fully supported yet, "
                                  f"use pandas RangeIndex instead")

    if not time_index.is_monotonic:
        raise ValueError("Time index must be monotonically increasing, but found non-monotonic index")

    return time_index


def validate_X(X):
    """Validate input data.

    Parameters
    ----------
    X : pandas DataFrame

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if X is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"`X` must a pandas DataFrame, but found: {type(X)}")
        if X.shape[0] > 1:
            raise ValueError(f"`X` must consist of a single row, but found: {X.shape[0]} rows")

        # Check if index is the same for all columns.

        # Get index from first row, can be either pd.Series or np.array.
        first_index = X.iloc[0, 0].index if hasattr(X.iloc[0, 0], 'index') else pd.RangeIndex(X.iloc[0, 0].shape[0])

        # Series must contain at least 2 observations, otherwise should be primitive.
        if len(first_index) < 1:
            raise ValueError(f'Time series must contain at least 2 observations, but found: '
                             f'{len(first_index)} observations in column: {X.columns[0]}')

        # Compare with remaining columns
        for c, col in enumerate(X.columns):
            index = X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index') else pd.RangeIndex(X.iloc[0, 0].shape[0])
            if not np.array_equal(first_index, index):
                raise ValueError(f'Found time series with unequal index in column {col}. '
                                 f'Input time-series must have the same index.')


def validate_sp(sp):
    """Validate seasonal periodicity.

    Parameters
    ----------
    sp : int
        Seasonal periodicity

    Returns
    -------
    sp : int
        Validated seasonal periodicity
    """

    if sp is None:
        return sp

    else:
        if not isinstance(sp, int) and (sp >= 0):
            raise ValueError(f"Seasonal periodicity (sp) has to be a positive integer, but found: "
                             f"{sp} of type: {type(sp)}")
        return sp


def validate_fh(fh):
    """Validate forecasting horizon.

    Parameters
    ----------
    fh : int or list of int
        Forecasting horizon with steps ahead to predict.

    Returns
    -------
    fh : numpy array of int
        Sorted and validated forecasting horizon.
    """
    # in-sample predictions
    if fh == "insample":
        return fh

    # Check single integer
    if np.issubdtype(type(fh), np.integer):
        fh = [fh]  # make iterable

    # Check array-like input
    elif isinstance(fh, list):
        if len(fh) < 1:
            raise ValueError(f"`fh` must specify at least one step, but found: "
                             f"{type(fh)} of length {len(fh)}")
        if not np.all([np.issubdtype(type(h), np.integer) for h in fh]):
            raise ValueError('If `fh` is passed as a list, '
                             'it has to be a list of integers')

    elif isinstance(fh, np.ndarray):
        if fh.ndim > 1:
            raise ValueError(f"`fh` must be a 1d array, but found: "
                             f"{fh.ndim} dimensions")
        if len(fh) < 1:
            raise ValueError(f"`fh` must specify at least one step, but found: "
                             f"{type(fh)} of length {len(fh)}")
        if not np.issubdtype(fh.dtype, np.integer):
            raise ValueError(
                f'If `fh` is passed as an array, it has to be an array of '
                f'integers, but found an array of dtype: {fh.dtype}')

    else:
        raise ValueError(f"`fh` has to be either a list or array of integers, or a single "
                         f"integer, but found: {type(fh)}")

    # check fh contains only non-zero positive values
    fh_sorted = np.sort(fh)
    if fh_sorted[0] <= 0:
        raise ValueError(f"fh must contain only positive values (> 0), but found: {fh}")

    return np.asarray(fh_sorted, dtype=np.int)


def check_is_fitted_in_transform(estimator, attributes, msg=None, all_or_any=all):
    """Checks if the estimator is fitted during transform by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    
    Raises
    ------
    NotFittedError
        If the attributes are not found.    
    """
    if msg is None:
        msg = ("This %(name)s instance has not been fitted yet. Call 'transform' with "
               "appropriate arguments before using this method.")

    check_is_fitted(estimator, attributes=attributes, msg=msg, all_or_any=all_or_any)


def check_consistent_time_indices(x, y):
    """Check that x and y have consistent indices.

    Parameters
    ----------
    x : pandas Series
    y : pandas Series

    Raises:
    -------
    ValueError
        If time indicies are not equal
    """

    if not x.index.equals(y.index):
        raise ValueError(f"Found input variables with inconsistent time indices")


def check_integer_time_index(time_index):
    if not np.issubdtype(time_index.dtype, np.dtype(int).type):
        raise NotImplementedError("Non-integer time indices are not supported yet")

    return time_index.values
