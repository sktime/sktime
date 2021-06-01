#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "CutoffSplitter",
    "SingleWindowSplitter",
    "temporal_train_test_split",
]
__author__ = ["Markus Löning", "Kutay Koralturk"]

import inspect
import numbers
import warnings
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import _pprint
from sklearn.model_selection import train_test_split as _train_test_split

from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_cutoffs
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_step_length
from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.validation.series import check_time_index

DEFAULT_STEP_LENGTH = 1
DEFAULT_WINDOW_LENGTH = 10
DEFAULT_FH = 1


def _repr(self):
    """Helper function to build repr for splitters similar to estimator objects"""
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


def _check_y(y):
    """Check input to `split` function"""
    if isinstance(y, pd.Series):
        y = y.index
    return check_time_index(y)


def _check_fh(fh):
    """Check and convert fh to format expected by CV splitters"""
    return check_fh(fh, enforce_relative=True)


def _get_end(y, fh):
    """Compute the end of the last training window for a given and forecasting
    horizon."""
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


def _check_window_lengths(y, fh, window_length, initial_window):
    n_timepoints = y.shape[0]
    fh_max = fh[-1]

    if window_length + fh_max > n_timepoints:
        raise ValueError(
            f"The `window_length` and the forecasting horizon are incompatible with "
            f"the length of `y`. Found `window_length`={window_length}, `max(fh)`="
            f"{fh_max}, but len(y)={n_timepoints}."
        )

    if initial_window is not None:
        if initial_window + fh_max > n_timepoints:
            raise ValueError(
                f"The `initial_window` and the forecasting horizon are incompatible "
                f"with the length of `y`. Found `initial_window`={initial_window},"
                f"`max(fh)`={fh_max}, but len(y)={n_timepoints}."
            )


class BaseSplitter:
    """Base class for temporal cross-validation splitters.

    Parameters
    ----------
    window_length : int
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(self, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH):
        self.window_length = window_length
        self.fh = fh

    def split(self, y):
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

    def _split(self, y):
        """Internal split method implemented by concrete classes"""
        raise NotImplementedError("abstract method")

    def get_n_splits(self, y=None):
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

    def get_cutoffs(self, y=None):
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

    def get_fh(self):
        """Return the forecasting horizon

        Returns
        -------
        fh : ForecastingHorizon
            The forecasting horizon
        """
        return check_fh(self.fh)

    def __repr__(self):
        return _repr(self)


class CutoffSplitter(BaseSplitter):
    """Cutoff window splitter.

    Split time series at given cutoff points into a fixed-length training and test set.

    Parameters
    ----------
    cutoffs : np.array
        cutoff points, positive and integer-index like, usable with pandas
        .iloc[] indexing
    fh : int, list or np.array
    window_length : int
    """

    def __init__(self, cutoffs, fh=DEFAULT_FH, window_length=DEFAULT_WINDOW_LENGTH):
        self.cutoffs = cutoffs
        super(CutoffSplitter, self).__init__(fh, window_length)

    def _split(self, y):
        # cutoffs
        cutoffs = check_cutoffs(self.cutoffs)
        if np.max(cutoffs) >= y.shape[0]:
            raise ValueError("`cutoffs` are incompatible with given `y`.")

        fh = _check_fh(self.fh)
        n_timepoints = y.shape[0]

        if np.max(cutoffs) + np.max(fh) > y.shape[0]:
            raise ValueError("`fh` is incompatible with given `cutoffs` and `y`.")
        window_length = check_window_length(self.window_length, n_timepoints)

        for cutoff in cutoffs:
            training_window = np.arange(cutoff - window_length, cutoff) + 1
            test_window = cutoff + fh
            yield training_window, test_window

    def get_n_splits(self, y=None):
        """Return the number of splits"""
        return len(self.cutoffs)

    def get_cutoffs(self, y=None):
        """Return the cutoff points"""
        return check_cutoffs(self.cutoffs)


class BaseWindowSplitter(BaseSplitter):
    """Base class for sliding and expanding window splitter"""

    def __init__(
        self,
        fh,
        initial_window,
        window_length,
        step_length,
        start_with_window,
    ):
        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super(BaseWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def _split(self, y):
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

            # For in-sample forecasting horizons, the first split must ensure that
            # in-sample test set is still within the data.
            if not fh.is_all_out_of_sample() and abs(fh[0]) >= self.initial_window:
                initial_start = abs(fh[0]) - self.initial_window + 1
            else:
                initial_start = 0

            initial_end = initial_start + initial_window
            train = np.arange(initial_start, initial_end)
            test = initial_end + fh.to_numpy() - 1
            yield train, test

        start = self._get_start(fh)
        end = _get_end(y, fh)

        for train, test in self._split_windows(
            start, end, step_length, window_length, fh.to_numpy()
        ):
            yield train, test

    @staticmethod
    def _split_windows(start, end, step_length, window_length, fh):
        """Abstract method implemented by concrete classes for sliding and expanding
        windows"""
        raise NotImplementedError("abstract method")

    def _get_start(self, fh):
        """Get the first split point"""
        # By default, the first split point is the index zero, the first
        # observation in
        # the data.
        start = 0

        # If we start with a full window, the first split point depends on the window
        # length.
        if hasattr(self, "start_with_window") and self.start_with_window:

            if hasattr(self, "initial_window") and self.initial_window is not None:

                if hasattr(self, "step_length"):
                    step_length = self.step_length
                else:
                    step_length = 1

                start += self.initial_window + step_length
            else:
                start += self.window_length

        # For in-sample forecasting horizons, the first split must ensure that
        # in-sample test set is still within the data.
        if not fh.is_all_out_of_sample():
            fh_min = abs(fh[0])
            if fh_min >= start:
                start = fh_min + 1

        return start

    def get_n_splits(self, y=None):
        """Return number of splits

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)

        Returns
        -------
        n_splits : int
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the "
                f"number of splits."
            )
        return len(self.get_cutoffs(y))

    def get_cutoffs(self, y=None):
        """Get the cutoff time points.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)

        Returns
        -------
        cutoffs : np.array
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        y = _check_y(y)
        fh = _check_fh(self.fh)
        step_length = check_step_length(self.step_length)

        if hasattr(self, "initial_window") and self.initial_window is not None:
            start = self.initial_window
        else:
            start = self._get_start(fh)

        end = _get_end(y, fh)

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
    window_length : int
        Window length
    step_length : int, optional (default=1)
        Step length between windows
    initial_window : int, optional (default=None)
        Window length of first window
    start_with_window : bool, optional (default=False)
        - If True, starts with full window.
        - If False, starts with empty window.
    """

    def __init__(
        self,
        fh=DEFAULT_FH,
        window_length=DEFAULT_WINDOW_LENGTH,
        step_length=DEFAULT_STEP_LENGTH,
        initial_window=None,
        start_with_window=True,
    ):
        super(SlidingWindowSplitter, self).__init__(
            fh=fh,
            window_length=window_length,
            initial_window=initial_window,
            step_length=step_length,
            start_with_window=start_with_window,
        )

    @staticmethod
    def _split_windows(start, end, step_length, window_length, fh):
        """Sliding windows"""
        for split_point in range(start, end, step_length):
            train = np.arange(split_point - window_length, split_point)
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
    initial_window : int, optional (default=10)
        Window length
    step_length : int, optional (default=1)
        Step length between windows
    start_with_window : bool, optional (default=False)
        - If True, starts with full window.
        - If False, starts with empty window.
    """

    def __init__(
        self,
        fh=DEFAULT_FH,
        initial_window=DEFAULT_WINDOW_LENGTH,
        step_length=DEFAULT_STEP_LENGTH,
        start_with_window=True,
    ):
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

    @staticmethod
    def _split_windows(start, end, step_length, window_length, fh):
        """Expanding windows"""
        for split_point in range(start, end, step_length):
            train = np.arange(start - window_length, split_point)
            test = split_point + fh - 1
            yield train, test


class SingleWindowSplitter(BaseSplitter):
    """Single window splitter.

    Split time series once into a training and test set.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int
        Window length
    """

    def __init__(self, fh, window_length=None):
        super(SingleWindowSplitter, self).__init__(fh, window_length)

    def _split(self, y):
        n_timepoints = y.shape[0]
        window_length = check_window_length(self.window_length, n_timepoints)
        fh = _check_fh(self.fh)

        end = _get_end(y, fh) - 1
        start = 0 if window_length is None else end - window_length
        train = np.arange(start, end)
        test = end + fh.to_numpy() - 1
        yield train, test

    def get_n_splits(self, y=None):
        """Return the number of splits.

        Parameters
        ----------
        y : pd.Series, optional (default=None)

        Returns
        -------
        n_splits : int
        """
        return 1

    def get_cutoffs(self, y=None):
        """Return the cutoff time points.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)

        Returns
        -------
        cutoffs : np.array
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        fh = _check_fh(self.fh)
        cutoff = _get_end(y, fh) - 2
        return np.array([cutoff])


def temporal_train_test_split(y, X=None, test_size=None, train_size=None, fh=None):
    """Split arrays or matrices into sequential train and test subsets
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
    splitting : tuple
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


def _split_by_fh(y, fh, X=None):
    """Helper function to split time series with forecasting horizon handling both
    relative and absolute horizons"""
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
