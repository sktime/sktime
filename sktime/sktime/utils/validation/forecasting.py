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
    "check_regressor",
]
__author__ = ["mloning", "big-o", "khrapovs"]

from datetime import timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.ensemble import GradientBoostingRegressor

from sktime.utils.validation import (
    array_is_datetime64,
    array_is_int,
    is_date_offset,
    is_int,
    is_timedelta,
)
from sktime.utils.validation.series import check_equal_time_index, check_series

ACCEPTED_CUTOFF_TYPES = list, np.ndarray, pd.Index
VALID_CUTOFF_TYPES = Union[ACCEPTED_CUTOFF_TYPES]


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
    """Check CV generators.

    Parameters
    ----------
    cv : CV generator

    Raises
    ------
    ValueError
        if cv does not have the required attributes.
    """
    from sktime.split.base import BaseSplitter

    if not isinstance(cv, BaseSplitter):
        raise TypeError(f"`cv` is not an instance of {BaseSplitter}")

    if enforce_start_with_window:
        if hasattr(cv, "start_with_window") and not cv.start_with_window:
            raise ValueError("`start_with_window` must be set to True")

    return cv


def check_step_length(step_length) -> Optional[int]:
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
    if step_length is None:
        return None

    elif is_int(step_length):
        if step_length < 1:
            raise ValueError(
                f"`step_length` must be a integer >= 1, " f"but found: {step_length}"
            )
        else:
            return step_length

    elif is_timedelta(step_length):
        if step_length <= timedelta(0):
            raise ValueError(
                f"`step_length` must be a positive timedelta, "
                f"but found: {step_length}"
            )
        else:
            return step_length

    elif is_date_offset(step_length):
        if step_length + pd.Timestamp(0) <= pd.Timestamp(0):
            raise ValueError(
                f"`step_length` must be a positive pd.DateOffset, "
                f"but found: {step_length}"
            )
        else:
            return step_length

    else:
        raise ValueError(
            f"`step_length` must be an integer, timedelta, pd.DateOffset, or None, "
            f"but found: {type(step_length)}"
        )


def check_sp(sp, enforce_list=False):
    """Validate seasonal periodicity.

    Parameters
    ----------
    sp : int or [int/float]
        Seasonal periodicity
    enforce_list : bool, optional (default=False)
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


def check_fh(fh, enforce_relative: bool = False, freq=None):
    """Coerce to ForecastingHorizon object and validate inputs.

    Parameters
    ----------
    fh : int, list, np.array, pd.Index or ForecastingHorizon
        Forecasting horizon specifying the time points to predict.
    enforce_relative : bool, optional (default=False)
        If True, checks if fh is relative.
    freq : str, or pd.Index, optional (default=None)
        object carrying frequency information on values
        ignored unless values is without inferable freq
        Frequency string or pd.Index

    Returns
    -------
    fh : ForecastingHorizon
        Validated forecasting horizon.

    Raises
    ------
    ValueError
        If passed fh is of length zero
        If enforce_relative is True, but fh.is_relative is False
    """
    # Convert to ForecastingHorizon
    from sktime.forecasting.base import ForecastingHorizon

    if not isinstance(fh, ForecastingHorizon):
        fh = ForecastingHorizon(fh, is_relative=None, freq=freq)
    else:
        fh.freq = freq

    # Check if non-empty, note we check for empty values here, rather than
    # during construction of ForecastingHorizon because ForecastingHorizon
    # can be empty in some cases, but users should not create forecasting horizons
    # with no values
    if len(fh) == 0:
        raise ValueError("`fh` must not be empty")

    if enforce_relative and not fh.is_relative:
        raise ValueError("`fh` must be relative, but found absolute `fh`")

    return fh


def check_alpha(alpha, name="alpha"):
    """Check that quantile or confidence level value, or list of values, is valid.

    Checks:
    alpha must be a float, or list of float, all in the open interval (0, 1).
    values in alpha must be unique.

    Parameters
    ----------
    alpha : float, list of float
    name : str, optional, default="alpha"
        the name reference to alpha displayed in the error message

    Returns
    -------
    alpha coerced to a list, i.e.: [alpha], if alpha was a float; alpha otherwise

    Raises
    ------
    ValueError
        If alpha (float) or any value in alpha (list) is outside the range (0, 1).
        If values in alpha (list) are non-unique.
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
                f"values in {name} must lie in the open interval (0, 1), "
                f"but found value: {a}."
            )

    # check uniqueness
    if len(set(alpha)) < len(alpha):
        raise ValueError(f"values in {name} must be unique, but found duplicates")

    return alpha


def check_cutoffs(cutoffs: VALID_CUTOFF_TYPES) -> np.ndarray:
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
    if not isinstance(cutoffs, ACCEPTED_CUTOFF_TYPES):
        raise ValueError(
            f"`cutoffs` must be a np.array or pd.Index, but found: {type(cutoffs)}"
        )
    assert array_is_int(cutoffs) or array_is_datetime64(cutoffs)

    if len(cutoffs) == 0:
        raise ValueError("Found empty `cutoff` array")

    return np.sort(cutoffs)


def check_scoring(scoring, allow_y_pred_benchmark=False, obj=None):
    """Validate a scorer parameter and coerce to sktime BaseMetric.

    Parameters
    ----------
    scoring : object to validate. For successful validation, must be one of

        * sktime metric object, instance of descendant of ``BaseMetric``
        * a callable with signature
          ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
          assuming `np.ndarray`-s being of the same length, and lower being better.
        * a string, resolvable by ``registry.resolve_alias`` to one of the above
        * None

    allow_y_pred_benchmark : boolean, optional, default=False
        whether to allow scorer classes
        with ``requires-y-pred-benchmark`` tag = ``True``

    obj : object or class, or None, optional, default=None
        if not None, will be used as a reference in the error message

    Returns
    -------
    scoring : input `scoring` coerced to instance of sktime `BaseMetric` descendant

        * if ``scoring`` was sktime metric, returns ``scoring``
        * if ``scoring`` was ``None``, returns ``MeanAbsolutePercentageError()``
        * if ``scoring`` was a callable, returns dynamic scoring metric class,
        as created by ``performance_metrics.forecasting.make_forecasting_scorer``

    Raises
    ------
    TypeError, if ``scoring`` is not a callable
    NotImplementedError
        if ``allow_y_pred_benchmark=False`` and metric
        requires ``y_pred_benchmark`` argument
    """
    # Note symmetric=True is default arg for MeanAbsolutePercentageError
    from sktime.performance_metrics.base import BaseMetric
    from sktime.performance_metrics.forecasting import (
        MeanAbsolutePercentageError,
        make_forecasting_scorer,
    )

    if scoring is None:
        return MeanAbsolutePercentageError()

    if obj is not None:
        obj_str = f" of {str(obj)}"
    else:
        obj_str = ""

    msg = (
        f"scoring parameter{obj_str} must be one of the following: "
        "(1) an sktime metric, descendant of BaseMetric; (2) a callable with signature "
        "(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float, "
        "assuming np.ndarrays being of the same length, and lower being better; "
        "(3) a string, resolvable by registry.resolve_alias to an object of either "
        "type (1) or (2)"
    )

    # deal with case (3) first - if string, try to resolve and return
    if isinstance(scoring, str):
        # lazy import of sktime.registry to avoid circular imports
        # and to ensure maximal decoupling from registry
        from sktime.registry import resolve_alias

        return resolve_alias(scoring)

    # check case (1) and case (2)
    # note: BaseMetric descendants are callable, so this is the same as
    # if not callable(scoring) and not isinstance(scoring, BaseMetric)
    if not callable(scoring):
        raise TypeError(msg)

    if not isinstance(scoring, BaseMetric):
        scoring = make_forecasting_scorer(func=scoring, greater_is_better=False)

    if hasattr(scoring, "get_class_tag"):
        scoring_req_bench = scoring.get_class_tag("requires-y-pred-benchmark", False)
        if scoring_req_bench and not allow_y_pred_benchmark:
            msg = (
                "Scoring requiring benchmark forecasts (y_pred_benchmark) are not "
                "fully supported yet. Please use a performance metric that does not "
                "require y_pred_benchmark as a keyword argument in its call signature."
            )
            raise NotImplementedError(msg)

    return scoring


def check_regressor(regressor=None, random_state=None):
    """Check if a valid regressor is given, otherwise set default regressor.

    Parameters
    ----------
    regressor : sklearn-like regressor, optional, default=None.
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.

    Returns
    -------
    regressor

    Raises
    ------
    ValueError
        Raise error if given regressor is not a valid sklearn-like regressor.
    """
    if regressor is None:
        regressor = GradientBoostingRegressor(max_depth=5, random_state=random_state)
    else:
        if not is_regressor(regressor):
            raise ValueError(
                f"`regressor` should be a sklearn-like regressor, "
                f"but found: {regressor}"
            )
        regressor = clone(regressor)
    return regressor


def check_interval_df(interval_df, index_to_match):
    """Verify that a predicted interval DataFrame is formatted correctly.

    Parameters
    ----------
    interval_df : pandas DataFrame outputted from forecaster.predict_interval()
    index_to_match : Index object that must match interval_df.index
    """
    from sktime.datatypes import check_is_mtype

    checked = check_is_mtype(
        interval_df, "pred_interval", return_metadata=True, msg_return_dict="list"
    )
    if not checked[0]:
        raise ValueError(checked[1])
    df_idx = interval_df.index
    if len(index_to_match) != len(df_idx) or not (index_to_match == df_idx).all():
        raise ValueError("Prediction interval index must match the final Series index.")
    levels = interval_df.columns.remove_unused_levels().levels
    if len(levels[0]) != 1:
        raise ValueError("`interval_df` must only contain one variable with interval")
