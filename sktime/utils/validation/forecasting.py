# -*- coding: utf-8 -*-

"""Validations for use with forecasting module."""

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

from sktime.utils import _has_tag
from sktime.utils.validation import is_int
from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.validation.series import check_series


def check_y_X(
    y,
    X=None,
    allow_empty=False,
    allow_constant=True,
    enforce_index_type=None,
):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series
    X : pd.DataFrame, optional (default=None)
    allow_empty : bool, optional (default=False)
        If True, empty `y` does not raise an error.
    allow_constant : bool, optional (default=True)
        If True, constant `y` does not raise an error.
    enforce_index_type : type, optional (default=None)
        type of time index

    Raises
    ------
    ValueError
        If y or X are invalid inputs
    """
    y = check_y(
        y,
        allow_empty=allow_empty,
        allow_constant=allow_constant,
        enforce_index_type=enforce_index_type,
    )

    if X is not None:
        # No need to also enforce the index type on X since we're
        # checking for index equality here
        X = check_X(X)
        check_equal_time_index(y, X)

    return y, X


def check_X(
    X,
    allow_empty=False,
    enforce_univariate=False,
    enforce_index_type=None,
):
    """Validate input data.

    Parameters
    ----------
    X : pd.Series, pd.DataFrame, np.ndarray
    allow_empty : bool, optional (default=False)
        If False, empty `X` raises an error.
    enforce_index_type : type, optional (default=None)
        type of time index
    enforce_univariate : bool, optional (default=False)
        If True, multivariate X will raise an error.

    Returns
    -------
    X : pd.Series, pd.DataFrame
        Validated input data.

    Raises
    ------
    ValueError, TypeError
        If X is an invalid input
    UserWarning
        Warning that X is given and model can't use it
    """
    # Check if pandas series or numpy array
    return check_series(
        X,
        enforce_univariate=enforce_univariate,
        allow_empty=allow_empty,
        enforce_index_type=enforce_index_type,
        allow_numpy=False,
    )


def check_y(y, allow_empty=False, allow_constant=True, enforce_index_type=None):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series
    allow_empty : bool, optional (default=False)
        If False, empty `y` raises an error.
    allow_constant : bool, optional (default=True)
        If True, constant `y` does not raise an error.
    enforce_index_type : type, optional (default=None)
        type of time index

    Returns
    -------
    y : pd.Series

    Raises
    ------
    ValueError, TypeError
        If y is an invalid input
    """
    y = check_series(
        y,
        enforce_univariate=True,
        allow_empty=allow_empty,
        allow_numpy=False,
        enforce_index_type=enforce_index_type,
    )

    if not allow_constant:
        if np.all(y == y.iloc[0]):
            raise ValueError("All values of `y` are the same.")

    return y


def check_cv(cv, enforce_start_with_window=False):
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

    if enforce_start_with_window:
        if hasattr(cv, "start_with_window") and not cv.start_with_window:
            raise ValueError("`start_with_window` must be set to True")

    return cv


def check_step_length(step_length):
    """Validate window length.

    Parameters
    ----------
    step_length : step length for data set.

    Returns
    -------
    step_length : int
        if step_length in not none and is int and greater than or equal to 1.

    Raises
    ------
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


def check_sp(sp, enforce_list=False):
    """Validate seasonal periodicity.

    Parameters
    ----------
    sp : int or [int/float]
        Seasonal periodicity
    emforce_list : bool, optional (default=False)
        If true, convert sp to list if not list.

    Returns
    -------
    sp : int or [int/float]
        Validated seasonal periodicity
    """
    if sp is not None:
        if enforce_list and is_int(sp) and sp >= 1:
            sp = [sp]
        elif (enforce_list and isinstance(sp, list)) or (is_int(sp) and sp >= 1):
            pass
        else:
            if enforce_list:
                raise ValueError("`sp` must be an int >= 1, [float/int] or None")
            else:
                raise ValueError("`sp` must be an int >= 1 or None")
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
        raise ValueError("`fh` must not be empty")

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
                "When `alpha` is passed as a list, it must be a list of floats"
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
    """Validate the cutoff.

    Parameters
    ----------
    cutoffs : np.ndarray or pd.Index

    Returns
    -------
    cutoffs (Sorted array)

    Raises
    ------
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


def check_scoring(scoring, allow_y_pred_benchmark=False):
    """
    Validate the performace scoring.

    Parameters
    ----------
    scoring : object that inherits from BaseMetric from sktime.performance_metrics.

    Returns
    -------
    scoring :
        MeanAbsolutePercentageError if the object is None.

    Raises
    ------
    TypeError
        if object is not callable from current scope.
    NotImplementedError
        if metric requires y_pred_benchmark to be passed
    """
    # Note symmetric=True is default arg for MeanAbsolutePercentageError
    from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

    if scoring is None:
        return MeanAbsolutePercentageError()

    if _has_tag(scoring, "requires-y-pred-benchmark") and not allow_y_pred_benchmark:
        msg = """Scoring requiring benchmark forecasts (y_pred_benchmark) are not
                 fully supported yet. Please use a performance metric that does not
                 require y_pred_benchmark as a keyword argument in its call signature.
              """
        raise NotImplementedError(msg)

    if not callable(scoring):
        raise TypeError("`scoring` must be a callable object")

    return scoring
