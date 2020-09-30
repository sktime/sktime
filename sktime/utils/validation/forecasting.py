__all__ = [
    "check_y",
    "check_X",
    "check_y_X",
    "check_fh",
    "check_cv",
    "check_window_length",
    "check_step_length",
    "check_time_index",
    "check_equal_time_index",
    "check_alpha",
    "check_fh_values",
    "check_cutoffs",
    "check_scoring",
    "check_sp",
    "check_fh_is_relative"
]
__author__ = ["Markus Löning", "@big-o"]

import numpy as np
import pandas as pd
from sktime.utils.validation import is_int


def check_y_X(y, X=None, allow_empty=False, allow_constant=True):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series
    X : pd.DataFrame, optional (default=None)
    allow_empty : bool, optional (default=False)
        If True, empty `y` does not raise an error.
    allow_constant : bool, optional (default=True)
        If True, constant `y` does not raise an error.

    Raises
    ------
    ValueError
        If y or X are invalid inputs
    """
    y = check_y(y, allow_empty=allow_empty, allow_constant=allow_constant)
    if X is not None:
        X = check_X(X)
        check_equal_time_index(y, X)
    return y, X


def check_y(y, allow_empty=False, allow_constant=True):
    """Validate input data.
    Parameters
    ----------
    y : pd.Series
    allow_empty : bool, optional (default=False)
        If True, empty `y` does not raise an error.
    allow_constant : bool, optional (default=True)
        If True, constant `y` does not raise an error.

    Returns
    -------
    y : pd.Series

    Raises
    ------
    ValueError, TypeError
        If y is an invalid input
    """
    # Check if pandas series or numpy array
    if not isinstance(y, pd.Series):
        raise TypeError(
            f"`y` must be a pandas Series, but found type: {type(y)}")

    # check that series is not empty
    if len(y) < 1:
        if not allow_empty:
            raise ValueError(
                f"`y` must contain at least some values, but found "
                f"empty series: {y}.")

    else:
        if not allow_constant:
            if np.all(y == y.iloc[0]):
                raise ValueError("All values of `y` are the same.")

    # check time index
    check_time_index(y.index)
    return y


def check_cv(cv):
    """
    Check CV generators.

    Parameters
    ----------
    cv : CV generator

    Raises
    ------
    ValueError
        if cv does not have the required attributes.
    """
    from sktime.forecasting.model_selection._split import BaseSplitter
    allowed_base_class = BaseSplitter
    if not isinstance(cv, allowed_base_class):
        raise TypeError(f"`cv` is not an instance of {allowed_base_class}")
    return cv


def check_time_index(time_index):
    """Check time index.

    Parameters
    ----------
    time_index : pd.Index or np.array

    Returns
    -------
    time_index : pd.Index
    """
    if isinstance(time_index, np.ndarray):
        time_index = pd.Index(time_index)

    # period or datetime index are not support yet
    supported_index_types = (pd.RangeIndex, pd.Int64Index, pd.UInt64Index)
    if not isinstance(time_index, supported_index_types):
        raise NotImplementedError(f"{type(time_index)} is not supported, "
                                  f"please use one of "
                                  f"{supported_index_types} instead.")

    if not time_index.is_monotonic:
        raise ValueError(
            f"The (time) index must be sorted (monotonically increasing), "
            f"but found: {time_index}")

    return time_index


def check_X(X):
    """Validate input data.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"`X` must a pd.DataFrame, but found: {type(X)}")
    return X


def check_window_length(window_length):
    """Validate window length"""
    if window_length is not None:
        if not is_int(window_length) or window_length < 1:
            raise ValueError(
                f"`window_length_` must be a positive integer >= 1 or None, "
                f"but found: {window_length}")
    return window_length


def check_step_length(step_length):
    """Validate window length"""
    if step_length is not None:
        if not is_int(step_length) or step_length < 1:
            raise ValueError(
                f"`step_length` must be a positive integer >= 1 or None, "
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
    fh : int, list of int, array of int or FH
        Forecasting horizon with steps ahead to predict.

    Returns
    -------
    fh : FH
        Checked forecasting horizon.
    """
    from sktime.forecasting.base._fh import FH
    if not isinstance(fh, FH):
        fh = FH(fh)
    return fh


def check_fh_is_relative(fh):
    """Check if fh is relative.

    Parameters
    ----------
    fh : np.array, list, int or FH

    Raises
    ------
    TypeError : if fh is not relative
    """
    from sktime.forecasting import FH
    if isinstance(fh, FH) and not fh.is_relative:
        raise TypeError("`fh` must be relative, but found absolute `fh`")


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


def check_cutoffs(cutoffs):
    if not isinstance(cutoffs, np.ndarray):
        raise ValueError(
            f"`cutoffs` must be a np.array, but found: {type(cutoffs)}")

    if not all([is_int(cutoff) for cutoff in cutoffs]):
        raise ValueError("All cutoff points must be integers")

    if not cutoffs.ndim == 1:
        raise ValueError("`cutoffs must be 1-dimensional array")

    if not len(cutoffs) > 0:
        raise ValueError("Found empty `cutoff` array")

    return np.sort(cutoffs)


def check_scoring(scoring):
    from sktime.performance_metrics.forecasting._classes import \
        MetricFunctionWrapper
    from sktime.performance_metrics.forecasting import sMAPE

    if scoring is None:
        return sMAPE()

    if not callable(scoring):
        raise TypeError("`scoring` must be a callable object")

    allowed_base_class = MetricFunctionWrapper
    if not isinstance(scoring, allowed_base_class):
        raise TypeError(
            f"`scoring` must inherit from `{allowed_base_class.__name__}`")

    return scoring


def check_fh_values(values):
    """Validate forecasting horizon values.

    Parameters
    ----------
    values : int, list of int, array of int
        Forecasting horizon with steps ahead to predict.

    Raises
    ------
    TypeError : if values do not meet criteria

    Returns
    -------
    fh : numpy array of int
        Sorted and validated forecasting horizon.
    """
    # check single integer
    if is_int(values):
        values = np.array([values], dtype=np.int)

    # check array
    elif isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise TypeError(f"`fh` must be a 1d array, but found shape: "
                            f"{values.shape}")

        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"If `fh` is passed as an array, it must "
                            f"be an array of integers, but found an "
                            f"array of type: {values.dtype}")

    # check list
    elif isinstance(values, list):
        if not np.all([is_int(h) for h in values]):
            raise TypeError("If `fh` is passed as a list, "
                            "it has to be a list of integers.")
        values = np.array(values, dtype=np.int)

    else:
        raise TypeError(f"`fh` has to be either a numpy array, list, "
                        f"or a single integer, but found: {type(values)}")

    # check fh is not empty
    if len(values) < 1:
        raise TypeError("`fh` cannot be empty, please specify now least one "
                        "step to forecast.")

    # check fh does not contain duplicates
    if len(values) != len(np.unique(values)):
        raise TypeError("`fh` should not contain duplicates.")

    # sort fh
    return np.sort(values)
