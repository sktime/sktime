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
__author__ = ["Markus Löning, Kutay Koralturk"]

import pandas as pd
import numpy as np

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


class BaseSplitter:
    """Base class for splitting time series during temporal cross-validation

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
        """Split y into windows.

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
        y = self._check_y(y)
        for training_window, test_window in self._split_windows(y):
            yield training_window[training_window >= 0], test_window[test_window >= 0]

    def _split_windows(self, y):
        """Internal split method"""
        raise NotImplementedError("abstract method")

    def get_n_splits(self, y=None):
        """Return the number of splits."""
        raise NotImplementedError("abstract method")

    def get_cutoffs(self, y=None):
        """Return the cutoff points in time at which y is split."""
        raise NotImplementedError("abstract method")

    def get_fh(self):
        """Return the forecasting horizon"""
        return check_fh(self.fh)

    @staticmethod
    def _check_y(y):
        # allow for pd.Series
        if isinstance(y, pd.Series):
            y = y.index
        return check_time_index(y)

    def _check_fh(self):
        return check_fh(self.fh, enforce_relative=True).to_numpy()


class CutoffSplitter(BaseSplitter):
    """Manual window splitter to split time series at given cutoff points.

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

    def _split_windows(self, y):
        # cutoffs
        cutoffs = check_cutoffs(self.cutoffs)
        if not np.max(cutoffs) < len(y):
            raise ValueError("`cutoffs` are out-of-bounds for given `y`.")

        fh = self._check_fh()

        if np.max(cutoffs) + np.max(fh) > len(y):
            raise ValueError("`fh` is out-of-bounds for given `cutoffs` and `y`.")
        window_length = check_window_length(self.window_length)

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
    """Base class for window splits"""

    def __init__(self, fh=None, window_length=None):
        super(BaseWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def _get_end(self, y):
        """Helper function to compute the end of the last window"""
        n_timepoints = len(y)
        fh = self._check_fh()
        window_length = check_window_length(self.window_length)
        # end point is end of last window
        is_in_sample = np.all(fh <= 0)
        if is_in_sample:
            end = n_timepoints + 1
        else:
            fh_max = fh[-1]
            end = n_timepoints - fh_max + 1  # non-inclusive end indexing

            # check if computed values are feasible with the provided index
            if window_length is not None:
                if window_length + fh_max > n_timepoints:
                    raise ValueError(
                        "The window length and forecasting horizon are "
                        "incompatible with the length of `y`"
                    )
        return end

    def split_initial(self, y):
        """Split initial window

        This is useful during forecasting model selection where we want to
        fit the forecaster on some part of the
        data first before doing temporal cross-validation

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        intial_training_window : np.array
        initial_test_window : np.array
        """
        if self.initial_window is None:
            raise ValueError(
                "Please specify initial window, found: `initial_window`=None"
            )
        initial = check_window_length(self.initial_window)
        initial_training_window = np.arange(initial)
        initial_test_window = np.arange(initial, len(y))
        return initial_training_window, initial_test_window


class SlidingWindowSplitter(BaseWindowSplitter):
    """Sliding window splitter

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int
    step_length : int
    initial_window : int
    start_with_window : bool, optional (default=False)

    Examples
    --------
    For example for `window_length = 5`, `step_length = 1` and `fh = 3`
    here is a representation of the folds::

    |-----------------------|
    | * * * * * x x x - - - |
    | - * * * * * x x x - - |
    | - - * * * * * x x x - |
    | - - - * * * * * x x x |


    ``*`` = training fold.

    ``x`` = test fold.
    """

    def __init__(
        self,
        fh=DEFAULT_FH,
        window_length=DEFAULT_WINDOW_LENGTH,
        step_length=DEFAULT_STEP_LENGTH,
        initial_window=None,
        start_with_window=False,
    ):

        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super(SlidingWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def _split_windows(self, y):
        step_length = check_step_length(self.step_length)
        window_length = check_window_length(self.window_length)
        fh = self._check_fh()

        end = self._get_end(y)
        start = self._get_start()
        for split_point in range(start, end, step_length):
            training_window = np.arange(split_point - window_length, split_point)
            test_window = split_point + fh - 1
            yield training_window, test_window

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
                f"{self.__class__.__name__} requires `y` to compute the " f"cutoffs."
            )
        y = self._check_y(y)
        end = self._get_end(y)
        start = self._get_start()
        step_length = check_step_length(self.step_length)
        return np.arange(start, end, step_length) - 1

    def _get_start(self):
        window_length = check_window_length(self.window_length)
        if self.start_with_window:
            return window_length
        else:
            return 0


class ExpandingWindowSplitter(BaseWindowSplitter):
    """Expanding window splitter

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int
    step_length : int
    initial_window : int
    start_with_window : bool, optional (default=False)

    Examples
    --------
    For example for `window_length = 5`, `step_length = 1` and `fh = 3`
    here is a representation of the folds::

    |-----------------------|
    | * * * * * x x x - - - |
    | * * * * * * x x x - - |
    | * * * * * * * x x x - |
    | * * * * * * * * x x x |


    ``*`` = training fold.

    ``x`` = test fold.
    """

    def __init__(
        self,
        fh=DEFAULT_FH,
        window_length=DEFAULT_WINDOW_LENGTH,
        step_length=DEFAULT_STEP_LENGTH,
        initial_window=None,
        start_with_window=False,
    ):

        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super(ExpandingWindowSplitter, self).__init__(
            fh=fh, window_length=window_length
        )

    def _split_windows(self, y):
        step_length = check_step_length(self.step_length)
        window_length = check_window_length(self.window_length)
        fh = self._check_fh()

        end = self._get_end(y)
        start = self._get_start()
        fixed_start = start
        for split_point in range(start, end, step_length):
            training_window = np.arange(fixed_start - window_length, split_point)
            test_window = split_point + fh - 1
            yield training_window, test_window

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
                f"{self.__class__.__name__} requires `y` to compute the " f"cutoffs."
            )
        y = self._check_y(y)
        end = self._get_end(y)
        start = self._get_start()
        step_length = check_step_length(self.step_length)
        return np.arange(start, end, step_length) - 1

    def _get_start(self):
        window_length = check_window_length(self.window_length)
        if self.start_with_window:
            return window_length
        else:
            return 0


class SingleWindowSplitter(BaseWindowSplitter):
    """Single window splitter

    Split time series once into a training and test window.

    Parameters
    ----------
    fh : int, list or np.array
    window_length : int
    """

    def __init__(self, fh, window_length=None):
        super(SingleWindowSplitter, self).__init__(fh, window_length)

    def _split_windows(self, y):
        window_length = check_window_length(self.window_length)
        fh = self._check_fh()

        end = self._get_end(y) - 1
        start = 0 if window_length is None else end - window_length
        training_window = np.arange(start, end)
        test_window = end + fh - 1
        yield training_window, test_window

    def get_n_splits(self, y=None):
        """Return number of splits

        Parameters
        ----------
        y : pd.Series, optional (default=None)

        Returns
        -------
        n_splits : int
        """
        return 1

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
                f"{self.__class__.__name__} requires `y` to compute the " f"cutoffs."
            )
        training_window, _ = next(self._split_windows(y))
        return training_window[-1:]  # array outpu

    def split_initial(self, y):
        """Split initial window

        This is useful during forecasting model selection where we want to
        fit the forecaster on some part of the
        data first before doing temporal cross-validation

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        intial_training_window : np.array
        initial_test_window : np.array
        """
        # the single window splitter simply returns the single split
        training_window, _ = next(self._split_windows(y))
        test_window = np.arange(training_window[-1] + 1, len(y))
        return training_window, test_window


def temporal_train_test_split(y, X=None, test_size=None, train_size=None, fh=None):
    """Split arrays or matrices into sequential train and test subsets
    Creates train/test splits over endogenous arrays an optional exogenous
    arrays. This is a wrapper of scikit-learn's ``train_test_split`` that
    does not shuffle.

    Parameters
    ----------
    *series : sequence of pd.Series with same length / shape[0]
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
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

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
        return _split_y_by_fh(y, fh, X=X)
    else:
        series = (y,) if X is None else (y, X)
        return _train_test_split(
            *series,
            shuffle=False,
            stratify=None,
            test_size=test_size,
            train_size=train_size,
        )


def _split_y_by_fh(y, fh, X=None):
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
