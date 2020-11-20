# -*- coding: utf-8 -*-
__all__ = [
    "check_y",
    "check_X",
    "check_y_X",
    "check_fh",
    "check_cv",
    "check_step_length",
    "check_alpha",
    "check_cutoffs",
    "check_scoring",
    "check_sp",
]
__author__ = ["Markus Löning", "@big-o"]

import numpy as np
import pandas as pd

from sktime.utils.validation import is_int
from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.validation.series import check_series


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


def check_X(X, allow_empty=False, enforce_univariate=False):
    """Validate input data.

    Parameters
    ----------
    X : pd.Series, pd.DataFrame, np.ndarray
    allow_empty : bool, optional (default=False)
        If True, empty `y` raises an error.

    Returns
    -------
    y : pd.Series, pd.DataFrame
        Validated input data.

    Raises
    ------
    ValueError, TypeError
        If y is an invalid input
    """
    # Check if pandas series or numpy array
    return check_series(
        X, enforce_univariate=enforce_univariate, allow_empty=allow_empty
    )


def check_y(y, allow_empty=False, allow_constant=True):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series
    allow_empty : bool, optional (default=False)
        If True, empty `y` raises an error.
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
    y = check_series(
        y, enforce_univariate=True, allow_empty=allow_empty, allow_numpy=False
    )

    if not allow_constant:
        if np.all(y == y.iloc[0]):
            raise ValueError("All values of `y` are the same.")

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

    if not isinstance(cv, BaseSplitter):
        raise TypeError(f"`cv` is not an instance of {BaseSplitter}")
    return cv


def check_step_length(step_length):
    """Validate window length.
    Parameters
    ----------
    step_length : step length for data set.

    Returns
    ----------
    step_length : int
        if step_length in not none and is int and greater than or equal to 1.

    Raises
    ----------
    ValueError
        if step_length is negative or not an integer or is None.
    """
    if step_length is not None:
        if not is_int(step_length) or step_length < 1:
            raise ValueError(
                f"`step_length` must be a positive integer >= 1 or None, "
                f"but found: {step_length}"
            )
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


def check_fh(fh, enforce_relative=False):
    """Validate forecasting horizon.

    Parameters
    ----------
    fh : int, list, np.array, pd.Index or ForecastingHorizon
        Forecasting horizon specifying the time points to predict.
    enforce_relative : bool, optional (default=False)
        If True, checks if fh is relative.

    Returns
    -------
    fh : ForecastingHorizon
        Validated forecasting horizon.
    """
    # Convert to ForecastingHorizon
    from sktime.forecasting.base import ForecastingHorizon

    if not isinstance(fh, ForecastingHorizon):
        fh = ForecastingHorizon(fh, is_relative=True)

    # Check if non-empty, note we check for empty values here, rather than
    # during construction of ForecastingHorizon because ForecastingHorizon
    # can be empty in some cases, but users should not create forecasting horizons
    # with no values
    if len(fh) == 0:
        raise ValueError(f"`fh` must not be empty, but found: {fh}")

    if enforce_relative and not fh.is_relative:
        raise ValueError("`fh` must be relative, but found absolute `fh`")

    return fh


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
            raise ValueError(
                "When `alpha` is passed as a list, " "it must be a list of floats"
            )

    elif isinstance(alpha, float):
        alpha = [alpha]  # make iterable

    # check range
    for a in alpha:
        if not 0 < a < 1:
            raise ValueError(
                f"`alpha` must lie in the open interval (0, 1), " f"but found: {a}."
            )

    return alpha


def check_cutoffs(cutoffs):
    """Validates the cutoff

    Parameters
    ----------
    cutoffs : np.ndarray or pd.Index

    Returns
    ----------
    cutoffs (Sorted array)

    Raises
    ----------
    ValueError
        If cutoffs is not a instance of np.array or pd.Index
        If cutoffs array is empty.

    """
    if not isinstance(cutoffs, (np.ndarray, pd.Index)):
        raise ValueError(
            f"`cutoffs` must be a np.array or pd.Index, " f"but found: {type(cutoffs)}"
        )
    assert np.issubdtype(cutoffs.dtype, np.integer)

    if len(cutoffs) == 0:
        raise ValueError("Found empty `cutoff` array")

    return np.sort(cutoffs)


def check_scoring(scoring):
    """
    Validates the performace scoring

    Parameters
    ----------
    scoring : object of class MetricFunctionWrapper from sktime.permormance_metrics.

    Returns
    ----------
    scoring : object of class MetricFunctionWrapper of sktime.permormance_metrics.
    sMAPE(mean percentage error)
        if the object is None.

    Raises
    ----------
    TypeError
        if object is not callable from current scope.
        if object is not an instance of class MetricFunctionWrapper of
        sktime.permormance_metrics.
    """
    from sktime.performance_metrics.forecasting._classes import MetricFunctionWrapper
    from sktime.performance_metrics.forecasting import sMAPE

    if scoring is None:
        return sMAPE()

    if not callable(scoring):
        raise TypeError("`scoring` must be a callable object")

    allowed_base_class = MetricFunctionWrapper
    if not isinstance(scoring, allowed_base_class):
        raise TypeError(f"`scoring` must inherit from `{allowed_base_class.__name__}`")

    return scoring
