#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement dataset splitting for model evaluation and selection."""

__all__ = [
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "CutoffSplitter",
    "SingleWindowSplitter",
    "temporal_train_test_split",
]
__author__ = ["mloning", "kkoralturk", "khrapovs"]

import inspect
import numbers
import warnings
from inspect import signature
from typing import Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import _pprint
from sklearn.model_selection import train_test_split as _train_test_split

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import VALID_FORECASTING_HORIZON_TYPES
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
    check_window_length,
    is_timedelta_or_date_offset,
)
from sktime.utils.validation.forecasting import (
    check_cutoffs,
    check_fh,
    check_step_length,
)
from sktime.utils.validation.series import check_equal_time_index, check_time_index

DEFAULT_STEP_LENGTH = 1
DEFAULT_WINDOW_LENGTH = 10
DEFAULT_FH = 1
ACCEPTED_Y_TYPES = Union[pd.Series, pd.DataFrame, np.ndarray, pd.Index]
FORECASTING_HORIZON_TYPES = Union[
    Union[VALID_FORECASTING_HORIZON_TYPES], ForecastingHorizon
]
SPLIT_TYPE = Union[
    Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]
]
SPLIT_GENERATOR_TYPE = Generator[Tuple[np.ndarray, np.ndarray], None, None]


def _repr(self) -> str:
    """Build repr for splitters similar to estimator objects."""
    # This is copied from scikit-learn's BaseEstimator get_params method
    cls = self.__class__
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, "cvargs"):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    def is_scalar_nan(x):
        return bool(isinstance(x, numbers.Real) and np.isnan(x))

    def has_changed(k, v):
        init_params = init_signature.parameters
        init_params = {name: param.default for name, param in init_params.items()}

        if k not in init_params:  # happens if k is part of a **kwargs
            return True
        if init_params[k] == inspect._empty:  # k has no default value
            return True

        # Use repr as a last resort. It may be expensive.
        if repr(v) != repr(init_params[k]) and not (
            is_scalar_nan(init_params[k]) and init_params(v)
        ):
            return True
        return False

    params = {k: v for k, v in params.items() if has_changed(k, v)}

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))


def _check_y(y: ACCEPTED_Y_TYPES) -> pd.Index:
    """Coerce input to `split` function.

    Parameters
    ----------
    y : pd.Series, pd.DataFrame, np.ndarray, or pd.Index
        coerced and checked version of input y

    Returns
    -------
    check_time_index(y_index), where y_index is as follows:
        if y is pd.Series, pd.DataFrame, y_index = y.index
        if y is pd.Index, y_index = y itself
        if y is np.ndarray, y_index =  pd.Index(y)

    Raises
    ------
    TypeError
        if y is not of one of the expected types
    NotImplementedError
        if y_index is not a supported sktime index type
    ValueError
        if y_index is not monotonous
    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_index = y.index
    elif isinstance(y, np.ndarray):
        y_index = pd.Index(y.flatten())
    elif isinstance(y, pd.Index):
        y_index = y
    else:
        raise TypeError(
            "Input to _check_y must be pd.Series, pd.DataFrame, np.ndarray, or pd.Index"
        )
    return check_time_index(index=y_index)


def _check_fh(fh: VALID_FORECASTING_HORIZON_TYPES) -> ForecastingHorizon:
    """Check and convert fh to format expected by CV splitters."""
    return check_fh(fh, enforce_relative=True)


def _get_end(y: ACCEPTED_Y_TYPES, fh: ForecastingHorizon) -> int:
    """Compute the end of the last training window for a forecasting horizon."""
    # `fh` is assumed to be ordered and checked by `_check_fh` and `window_length` by
    # `check_window_length`.
    n_timepoints = y.shape[0]

    # For purely in-sample forecasting horizons, the last split point is the end of the
    # training data.
    if fh.is_all_in_sample():
        end = n_timepoints + 1

    # Otherwise, the last point must ensure that the last horizon is within the data.
    else:
        fh_max = fh[-1]
        end = n_timepoints - fh_max + 1

    return end


def _check_window_lengths(
    y: ACCEPTED_Y_TYPES,
    fh: ForecastingHorizon,
    window_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
    initial_window: NON_FLOAT_WINDOW_LENGTH_TYPES,
) -> None:
    n_timepoints = y.shape[0]
    fh_max = fh[-1]

    error_msg_for_incompatible_window_length = (
        f"The `window_length` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `window_length`={window_length}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the window length plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    if is_timedelta_or_date_offset(x=window_length):
        if y.get_loc(min(y[-1], y[0] + window_length)) + fh_max > n_timepoints:
            raise ValueError(error_msg_for_incompatible_window_length)
    else:
        if window_length + fh_max > n_timepoints:
            raise ValueError(error_msg_for_incompatible_window_length)

    error_msg_for_incompatible_initial_window = (
        f"The `initial_window` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `initial_window`={initial_window}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the initial window plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    error_msg_for_incompatible_types = (
        "The `initial_window` and `window_length` types are incompatible. "
        "They should be either all timedelta or all int."
    )
    if initial_window is not None:
        if is_timedelta_or_date_offset(x=initial_window):
            if y.get_loc(min(y[-1], y[0] + initial_window)) + fh_max > n_timepoints:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if not is_timedelta_or_date_offset(x=window_length):
                raise ValueError(error_msg_for_incompatible_types)
        else:
            if initial_window + fh_max > n_timepoints:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if is_timedelta_or_date_offset(x=window_length):
                raise ValueError(error_msg_for_incompatible_types)


class BaseSplitter:
    r"""Base class for temporal cross-validation splitters.

    The purpose of this implementation is to fill the gap relative to
    `sklearn.model_selection.TimeSeriesSplit
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>`__
    which implements only expanding window split strategy, and only integer based.

    The most important method in this class is `.split(y)` which generates indices
    of non-overlapping train/test splits of a time series `y`.
    The length of the train split is determined by `window_length`.
    The length of the test split is determined by forecasting horizon `fh`.

    In general, splitting a time series :math:`y=(y_1,\ldots,y_T)`
    into train/test splits means separating it into two non-overlapping series:
    train :math:`(y_{t(1)},\ldots,y_{t(k)})`
    and test :math:`(y_{t(k+1)},\ldots,y_{t(k+l)})`,
    where :math:`k,l` are all integers greater than zero,
    and :math:`t(k)<t(k+1)` are ordered time indices.
    The exact set of indices depends on a concrete splitter.
    Method `.split` is used to generate a pair of index sets:
    train :math:`\{t(1),\ldots,t(k)\}` and test :math:`\{t(k+1),\ldots,t(k+l)\}`.

    In case `window_length` and `fh` are integer valued,
    they translate into :math:`k` and :math:`l`, respectively.

    In case `window_length` and `fh` can be interpreted
    as time interval length (time deltas), then they correspond to
    :math:`t(k)-t(1)` and :math:`t(k+l)-t(k+1)`, respectively.

    Method `.get_n_splits` returns the number of splitting iterations.
    This number depends on a concrete splitting strategy and splitter parameters.

    Method `.get_cutoffs` returns the cutoff points between each train/test split.
    Using the above notation, for a single split it corresponds
    to the last index of the training window, :math:`t(k)`

    In order to illustrate the difference in integer/interval arithmetic
    in calculating train/test indices, let us consider the following examples.
    Suppose, the arguments of a splitter are `cutoff = 10` and `window_length = 6`.
    Then, we have `train_start = cutoff - window_length = 4`.
    For timedelta-like values the logic is a bit more complicated.
    The time point corresponding to the `cutoff`
    (index value of the `y` series) is shifted back
    by the timedelta `window_length`,
    and then the integer position of the resulting datetime
    is considered to be the training window start.
    For example, for `cutoff = 10`, and `window_length = pd.Timedelta(6, unit="D")`,
    we have `y[cutoff] = pd.Timestamp("2021-01-10")`,
    and `y[cutoff] - window_length = pd.Timestamp("2021-01-04")`,
    which leads to `train_start = y.loc(y[cutoff] - window_length) = 4`.
    Similar timedelta arithmetic applies to other splitter arguments.

    Parameters
    ----------
    window_length : int or timedelta or pd.DateOffset
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.

    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
    ) -> None:
        self.window_length = window_length
        self.fh = fh

    def split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        """Split `y` into training and test windows.

        Parameters
        ----------
        y : pd.Series or pd.Index
            Time series to split

        Yields
        ------
        train : np.array
            Training window indices
        test : np.array
            Test window indices
        """
        y = _check_y(y)
        for train, test in self._split(y):
            yield train[train >= 0], test[test >= 0]

    def _split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        """Split `y` into training and test windows.

        Parameters
        ----------
        y : pd.Series or pd.Index
            Time series to split

        Yields
        ------
        training_window : np.array
            Training window indices
        test_window : np.array
            Test window indices
        """
        raise NotImplementedError("abstract method")

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        raise NotImplementedError("abstract method")

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : np.array
            The array of cutoff points.
        """
        raise NotImplementedError("abstract method")

    def get_fh(self) -> ForecastingHorizon:
        """Return the forecasting horizon.

        Returns
        -------
        fh : ForecastingHorizon
            The forecasting horizon
        """
        return check_fh(self.fh)

    def __repr__(self) -> str:
        return _repr(self)


class CutoffSplitter(BaseSplitter):
    r"""Cutoff window splitter.

    Split time series at given cutoff points into a fixed-length training and test set.

    Here the user is expected to provide a set of cutoffs (train set endpoints),
    which using the notation provided in :class:`BaseSplitter`,
    can be written as :math:`\{t(k_1),\ldots,t(k_n)\}`.
    The number of splits returned by `.get_n_splits`
    is then trivially equal to :math:`n`.
    The sorted array of cutoffs returned by `.get_cutoffs` is then equal to
    :math:`\{t(k_1),\ldots,t(k_n)\}` with :math:`k_i<k_{i+1}`.

    Parameters
    ----------
    cutoffs : np.array or pd.Index
        cutoff points, positive and integer-index like, usable with pandas
        .iloc[] indexing
    fh : int, list or np.array
    window_length : int or timedelta or pd.DateOffset
    """

    def __init__(
        self,
        cutoffs: Union[np.ndarray, pd.Index],
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
    ) -> None:
        self.cutoffs = cutoffs
        super(CutoffSplitter, self).__init__(fh, window_length)

    def _split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        cutoffs = check_cutoffs(self.cutoffs)
        if np.max(cutoffs) >= y.shape[0]:
            raise ValueError("`cutoffs` are incompatible with given `y`.")

        fh = _check_fh(self.fh)
        n_timepoints = y.shape[0]

        if np.max(cutoffs) + np.max(fh) > y.shape[0]:
            raise ValueError("`fh` is incompatible with given `cutoffs` and `y`.")
        window_length = check_window_length(self.window_length, n_timepoints)
        for cutoff in cutoffs:
            if is_timedelta_or_date_offset(x=window_length):
                train_start = y.get_loc(max(y[0], y[cutoff] - window_length))
            else:
                train_start = cutoff - window_length
            training_window = np.arange(train_start, cutoff) + 1
            test_window = cutoff + fh
            yield training_window, test_window

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        For this splitter the number is trivially equal to
        the number of cutoffs given during instance initialization.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return len(self.cutoffs)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points.

        This method trivially returns the cutoffs given during instance initialization.
        The only change is that the set of cutoffs is sorted from smallest to largest.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : np.array
            The array of cutoff points.
        """
        return check_cutoffs(self.cutoffs)


class BaseWindowSplitter(BaseSplitter):
    """Base class for sliding and expanding window splitter."""

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
        start_with_window: bool,
    ) -> None:
        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super(BaseWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def _split(self, y: Optional[ACCEPTED_Y_TYPES]) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        step_length = check_step_length(self.step_length)
        window_length = check_window_length(
            self.window_length, n_timepoints, "window_length"
        )
        initial_window = check_window_length(
            self.initial_window, n_timepoints, "initial_window"
        )
        fh = _check_fh(self.fh)
        _check_window_lengths(y, fh, window_length, initial_window)

        if self.initial_window is not None:
            if not self.start_with_window:
                raise ValueError(
                    "`start_with_window` must be True if `initial_window` is given"
                )

            if self.initial_window <= self.window_length:
                raise ValueError("`initial_window` must greater than `window_length`")

            if is_timedelta_or_date_offset(x=self.initial_window):
                initial_window_threshold = y.get_loc(y[0] + self.initial_window)
            else:
                initial_window_threshold = self.initial_window
            # For in-sample forecasting horizons, the first split must ensure that
            # in-sample test set is still within the data.
            if not fh.is_all_out_of_sample() and abs(fh[0]) >= initial_window_threshold:
                initial_start = abs(fh[0]) - self.initial_window + 1
            else:
                initial_start = 0

            if is_timedelta_or_date_offset(x=initial_window):
                initial_end = y.get_loc(y[initial_start] + initial_window)
            else:
                initial_end = initial_start + initial_window
            train = np.arange(initial_start, initial_end)
            test = initial_end + fh.to_numpy() - 1
            yield train, test

        start = self._get_start(y=y, fh=fh)
        end = _get_end(y=y, fh=fh)
        step_length = self._get_step_length(x=step_length)

        for train, test in self._split_windows(
            start, end, step_length, window_length, y, fh.to_numpy()
        ):
            yield train, test

    def _split_windows(
        self,
        start: int,
        end: int,
        step_length: int,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: ACCEPTED_Y_TYPES,
        fh: np.ndarray,
    ) -> SPLIT_GENERATOR_TYPE:
        """Abstract method for sliding/expanding windows."""
        raise NotImplementedError("abstract method")

    @staticmethod
    def _get_train_start(
        start, window_length: ACCEPTED_WINDOW_LENGTH_TYPES, y: ACCEPTED_Y_TYPES
    ) -> int:
        if is_timedelta_or_date_offset(x=window_length):
            train_start = y.get_loc(
                max(y[min(start, len(y) - 1)] - window_length, min(y))
            )
            if start >= len(y):
                train_start += 1
        else:
            train_start = start - window_length
        return train_start

    def _get_start(self, y: ACCEPTED_Y_TYPES, fh: ForecastingHorizon) -> int:
        """Get the first split point."""
        # By default, the first split point is the index zero, the first
        # observation in
        # the data.
        start = 0

        # If we start with a full window, the first split point depends on the window
        # length.
        if hasattr(self, "start_with_window") and self.start_with_window:

            if hasattr(self, "initial_window") and self.initial_window is not None:

                if hasattr(self, "step_length"):
                    step_length = self._get_step_length(x=self.step_length)
                else:
                    step_length = 1

                if is_timedelta_or_date_offset(x=self.initial_window):
                    start = y.get_loc(y[start] + self.initial_window) + step_length
                else:
                    start += self.initial_window + step_length
            else:
                if is_timedelta_or_date_offset(x=self.window_length):
                    start = y.get_loc(y[start] + self.window_length)
                else:
                    start += self.window_length

        # For in-sample forecasting horizons, the first split must ensure that
        # in-sample test set is still within the data.
        if not fh.is_all_out_of_sample():
            fh_min = abs(fh[0])
            if fh_min >= start:
                start = fh_min + 1

        return start

    @staticmethod
    def _get_step_length(x: NON_FLOAT_WINDOW_LENGTH_TYPES) -> int:
        return _coerce_duration_to_int(duration=x, freq="D")

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the "
                f"number of splits."
            )
        return len(self.get_cutoffs(y))

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : np.array
            The array of cutoff points.
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        y = _check_y(y)
        fh = _check_fh(self.fh)
        step_length = check_step_length(self.step_length)

        if hasattr(self, "initial_window") and self.initial_window is not None:
            if is_timedelta_or_date_offset(x=self.initial_window):
                start = y.get_loc(y[0] + self.initial_window)
            else:
                start = self.initial_window
        else:
            start = self._get_start(y=y, fh=fh)

        end = _get_end(y, fh)
        step_length = self._get_step_length(x=step_length)

        return np.arange(start, end, step_length) - 1


class SlidingWindowSplitter(BaseWindowSplitter):
    """Sliding window splitter.

    Split time series repeatedly into a fixed-length training and test set.

    For example for `window_length = 5`, `step_length = 1` and `fh = 3`
    here is a representation of the folds::

    |-----------------------|
    | * * * * * x x x - - - |
    | - * * * * * x x x - - |
    | - - * * * * * x x x - |
    | - - - * * * * * x x x |


    ``*`` = training fold.

    ``x`` = test fold.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int or timedelta or pd.DateOffset
        Window length
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    initial_window : int or timedelta or pd.DateOffset, optional (default=None)
        Window length of first window
    start_with_window : bool, optional (default=False)
        - If True, starts with full window.
        - If False, starts with empty window.
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
        start_with_window: bool = True,
    ) -> None:
        super(SlidingWindowSplitter, self).__init__(
            fh=fh,
            window_length=window_length,
            initial_window=initial_window,
            step_length=step_length,
            start_with_window=start_with_window,
        )

    def _split_windows(
        self,
        start: int,
        end: int,
        step_length: int,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: ACCEPTED_Y_TYPES,
        fh: np.ndarray,
    ) -> SPLIT_GENERATOR_TYPE:
        for split_point in range(start, end, step_length):
            train_start = self._get_train_start(
                start=split_point, window_length=window_length, y=y
            )
            train = np.arange(train_start, split_point)
            test = split_point + fh - 1
            yield train, test


class ExpandingWindowSplitter(BaseWindowSplitter):
    """Expanding window splitter.

    Split time series repeatedly into an growing training set and a fixed-size test set.

    For example for `window_length = 5`, `step_length = 1` and `fh = 3`
    here is a representation of the folds::

    |-----------------------|
    | * * * * * x x x - - - |
    | * * * * * * x x x - - |
    | * * * * * * * x x x - |
    | * * * * * * * * x x x |


    ``*`` = training fold.

    ``x`` = test fold.

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon
    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Window length
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    start_with_window : bool, optional (default=False)
        - If True, starts with full window.
        - If False, starts with empty window.
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        start_with_window: bool = True,
    ) -> None:
        # Note that we pass the initial window as the window_length below. This
        # allows us to use the common logic from the parent class, while at the same
        # time expose the more intuitive name for the ExpandingWindowSplitter.
        super(ExpandingWindowSplitter, self).__init__(
            fh=fh,
            window_length=initial_window,
            initial_window=None,
            step_length=step_length,
            start_with_window=start_with_window,
        )

    def _split_windows(
        self,
        start: int,
        end: int,
        step_length: int,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: ACCEPTED_Y_TYPES,
        fh: np.ndarray,
    ) -> SPLIT_GENERATOR_TYPE:
        for split_point in range(start, end, step_length):
            train_start = self._get_train_start(
                start=start, window_length=window_length, y=y
            )
            train = np.arange(train_start, split_point)
            test = split_point + fh - 1
            yield train, test


class SingleWindowSplitter(BaseSplitter):
    """Single window splitter.

    Split time series once into a training and test set.
    See more details on what to expect from this splitter in :class:`BaseSplitter`.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int or timedelta or pd.DateOffset
        Window length
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        window_length: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
    ) -> None:
        super(SingleWindowSplitter, self).__init__(fh, window_length)

    def _split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        window_length = check_window_length(self.window_length, n_timepoints)
        fh = _check_fh(self.fh)

        end = _get_end(y, fh) - 1
        if window_length is None:
            start = 0
        elif is_timedelta_or_date_offset(x=window_length):
            start = y.get_loc(y[end - 1] - window_length) + 1
        else:
            start = end - window_length
        train = np.arange(start, end)
        test = end + fh.to_numpy() - 1
        yield train, test

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        Since this splitter returns a single train/test split,
        this number is trivially 1.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return 1

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points.

        Since this splitter returns a single train/test split,
        this method returns a single one-dimensional array
        with the last train set index.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : np.array
            The array of cutoff points.
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        fh = _check_fh(self.fh)
        cutoff = _get_end(y, fh) - 2
        return np.array([cutoff])


def temporal_train_test_split(
    y: ACCEPTED_Y_TYPES,
    X: Optional[pd.DataFrame] = None,
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
    fh: Optional[FORECASTING_HORIZON_TYPES] = None,
) -> SPLIT_TYPE:
    """Split arrays or matrices into sequential train and test subsets.

    Creates train/test splits over endogenous arrays an optional exogenous
    arrays.

    This is a wrapper of scikit-learn's ``train_test_split`` that
    does not shuffle the data.

    Parameters
    ----------
    y : pd.Series
        Target series
    X : pd.DataFrame, optional (default=None)
        Exogenous data
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        relative number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the relative number of train samples. If None,
        the value is automatically set to the complement of the test size.
    fh : ForecastingHorizon

    Returns
    -------
    splitting : tuple, length=2 * len(arrays)
        List containing train-test split of `y` and `X` if given.

    References
    ----------
    ..[1]  adapted from https://github.com/alkaline-ml/pmdarima/
    """
    if fh is not None:
        if test_size is not None or train_size is not None:
            raise ValueError(
                "If `fh` is given, `test_size` and `train_size` cannot "
                "also be specified."
            )
        return _split_by_fh(y, fh, X=X)
    else:
        series = (y,) if X is None else (y, X)
        return _train_test_split(
            *series,
            shuffle=False,
            stratify=None,
            test_size=test_size,
            train_size=train_size,
        )


def _split_by_fh(
    y: ACCEPTED_Y_TYPES, fh: FORECASTING_HORIZON_TYPES, X: Optional[pd.DataFrame] = None
) -> SPLIT_TYPE:
    """Split time series with forecasting horizon.

    Handles both relative and absolute horizons.
    """
    if X is not None:
        check_equal_time_index(y, X)
    fh = check_fh(fh)
    idx = fh.to_pandas()
    index = y.index

    if fh.is_relative:
        if not fh.is_all_out_of_sample():
            raise ValueError("`fh` must only contain out-of-sample values")
        max_step = idx.max()
        steps = fh.to_indexer()
        train = index[:-max_step]
        test = index[-max_step:]

        y_test = y.loc[test[steps]]

    else:
        min_step, max_step = idx.min(), idx.max()
        train = index[index < min_step]
        test = index[(index <= max_step) & (min_step <= index)]

        y_test = y.loc[idx]

    y_train = y.loc[train]
    if X is None:
        return y_train, y_test

    else:
        X_train = X.loc[train]
        X_test = X.loc[test]
        return y_train, y_test, X_train, X_test
