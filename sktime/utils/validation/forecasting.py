__all__ = [
    "check_y",
    "check_X",
    "check_y_X",
    "check_fh",
    "check_cv",
    "check_time_index",
    "check_consistent_time_index",
    "check_alpha",
    "check_is_fitted_in_transform"
]
__author__ = ["Markus Löning", "@big-o"]

import numpy as np
import pandas as pd

from sktime.utils.validation import check_is_fitted
from sktime.utils.validation import is_int


def check_y_X(y, X):
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
    y = check_y(y)
    X = check_X(X)
    return y, X


def check_y(y):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series or np.ndarray

    Returns
    -------
    y : pd.Series

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    # Check if pandas series or numpy array
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(f"`y` must be a pandas Series or numpy array, but found: {type(y)}")

    # if numpy array, transform into series to handle time index
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            raise ValueError(f"`y` must be 1-dimensional, but found series of shape: {y.shape}")
        y = pd.Series(y)

    # check that series is not empty
    if y.size < 1:
        raise ValueError(f"`y` must contain at least some observations, but found empty series: {y}")

    # check time index
    check_time_index(y.index)
    return y


def check_cv(cv):
    if not hasattr(cv, "split"):
        raise ValueError("Expected cv as a temporal cross-validation object with `split` method")

    if not hasattr(cv, "fh"):
        raise ValueError("Expected cv as a temporal cross-validation object with `fh` attribute")

    if not hasattr(cv, "window_length"):
        raise ValueError("Expected cv as a temporal cross-validation object with `window_length` attribute")

    if not hasattr(cv, "step_length"):
        raise ValueError("Expected cv as a temporal cross-validation object with `step_length` attribute")

    return cv


def check_time_index(time_index):
    """Validate time index

    Parameters
    ----------
    time_index : pd.Index

    Returns
    -------
    time_index : pd.Index
    """
    # input conversion
    if isinstance(time_index, pd.Series):
        time_index = time_index.index  # get pandas index

    elif isinstance(time_index, np.ndarray):
        if not time_index.ndim == 1:
            raise ValueError("Cannot construct time index from multi-dimensional numpy array; "
                             "please pass a 1d array as the time index.")
        time_index = pd.Index(time_index)

    # input checks
    if isinstance(time_index, pd.Index):
        # period or datetime index are not support yet
        supported_index_types = (pd.RangeIndex, pd.Int64Index, pd.UInt64Index)
        if not isinstance(time_index, supported_index_types):
            raise NotImplementedError(f"{type(time_index)} is not supported yet, "
                                      f"please use one of {supported_index_types} instead.")

    if not time_index.is_monotonic:
        raise ValueError(f"Time index must be sorted (monotonically increasing), "
                         f"but found: {time_index}")

    return time_index


def check_X(X):
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

    return X


def check_window_length(window_length):
    """Validate window length"""
    if window_length is not None:
        if not is_int(window_length) or window_length < 1:
            raise ValueError(f"`window_length` must be a positive integer >= 1 or None, "
                             f"but found: {window_length}")
    return window_length


def check_step_length(step_length):
    """Validate window length"""
    if step_length is not None:
        if not is_int(step_length) or step_length < 1:
            raise ValueError(f"`step_length` must be a positive integer >= 1 or None, "
                             f"but found: {step_length}")
    return step_length


def check_sp(sp):
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
    if sp is not None:
        if not is_int(sp) or sp < 1:
            raise ValueError("`sp` must be a positive integer >= 1 or None")
    return sp


def check_fh(fh):
    """Validate forecasting horizon.

    Parameters
    ----------
    fh : int, list of int, array of int
        Forecasting horizon with steps ahead to predict.

    Returns
    -------
    fh : numpy array of int
        Sorted and validated forecasting horizon.
    """
    # check single integer
    if is_int(fh):
        fh = np.array([fh], dtype=np.int)

    # check array input
    elif isinstance(fh, np.ndarray):
        if fh.ndim > 1:
            raise ValueError(f"`fh` must be a 1d array, but found shape: "
                             f"{fh.shape}")

        if not np.issubdtype(fh.dtype, np.integer):
            raise ValueError(f"If `fh` is passed as an array, it must "
                             f"be an array of integers, but found an "
                             f"array of type: {fh.dtype}")

    # check list input
    elif isinstance(fh, list):
        if not np.all([is_int(h) for h in fh]):
            raise ValueError("If `fh` is passed as a list, "
                             "it has to be a list of integers.")
        fh = np.array(fh, dtype=np.int)

    else:
        raise ValueError(f"`fh` has to be either a numpy array or list of integers, "
                         f"or a single integer, but found: {type(fh)}")

    # check fh is not empty
    if len(fh) < 1:
        raise ValueError(f"`fh` cannot be empty, please specify at least one "
                         f"step to forecast.")

    # sort fh
    fh.sort()

    # check fh contains only non-zero positive values
    if fh[0] <= 0:
        raise ValueError(f"fh must contain only positive values (>=1), "
                         f"but found: {fh[0]}")

    return fh


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


def check_consistent_time_index(y_test, y_pred, y_train=None):
    """Check that y_test and y_pred have consistent indices.

    Parameters
    ----------
    y_test : pd.Series
    y_pred : pd.Series
    y_train : pd.Series

    Raises
    ------
    ValueError
        If time indicies are not equal
    """

    # only validate indices if data is passed as pd.Series
    check_time_index(y_test.index)
    check_time_index(y_pred.index)

    if not y_test.index.equals(y_pred.index):
        print(y_test.index, y_pred.index)
        raise ValueError(f"Time index of `y_pred` does not match time index of `y_test`.")

    if y_train is not None:
        check_time_index(y_train.index)
        if y_train.index.max() >= y_pred.index.min():
            raise ValueError(f"Found `y_train` with time index which is not "
                             f"before time index of `y_pred`")


def check_alpha(alpha):
    """Check that a confidence level alpha (or list of alphas) is valid.

    All alpha values must lie in the open interval (0, 1).

    Parameters
    ----------
    alpha : float, list of float

    Raises
    ------
    ValueError
        If alpha is outside the range (0, 1).
    """
    # check type
    if isinstance(alpha, list):
        if not all(isinstance(a, float) for a in alpha):
            raise ValueError("When `alpha` is passed as a list, "
                             "it must be a list of floats")

    elif isinstance(alpha, float):
        alpha = [alpha]  # make iterable

    # check range
    for a in alpha:
        if not 0 < a < 1:
            raise ValueError(f"`alpha` must lie in the open interval (0, 1), "
                             f"but found: {a}.")

    return alpha
