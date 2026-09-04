#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that modifies a splitter by adding train folds to the test folds."""

__author__ = ["fkiraly"]

__all__ = ["TestPlusTrainSplitter"]


import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter


class TestPlusTrainSplitter(BaseSplitter):
    r"""Splitter that adds the train sets to the test sets.

    Takes a splitter ``cv`` and modifies it in the following way:
    The i-th train sets is identical to the i-th train set of ``cv``.
    The i-th test set is the union of the i-th train set and i-th test set of ``cv``.

    Parameters
    ----------
    cv : BaseSplitter
        splitter to modify as above

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import ExpandingWindowSplitter, TestPlusTrainSplitter

    >>> y = load_airline()
    >>> y_template = y[:60]
    >>> cv_tpl = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)

    >>> splitter = TestPlusTrainSplitter(cv_tpl)
    """

    def __init__(self, cv):
        self.cv = cv
        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        # dispatch split_series to the same split/split_loc as the wrapped cv
        # for performance reasons
        self.clone_tags(self.cv, "split_series_uses")

    def _split(self, y: pd.Index):
        """Get iloc references to train/test splits of ``y``.

        private _split containing the core logic, called from split

        Parameters
        ----------
        y : pd.Index or time series in sktime compatible time series format
            Time series to split, or index of time series to split

        Yields
        ------
        train : 1D np.ndarray of dtype int
            Training window indices, iloc references to training indices in y
        test : 1D np.ndarray of dtype int
            Test window indices, iloc references to test indices in y
        """
        cv = self.cv

        for y_train_inner, y_test_inner in cv.split(y):
            y_train_self = y_train_inner
            y_test_self = np.union1d(y_train_inner, y_test_inner)
            yield y_train_self, y_test_self

    def _split_loc(self, y: pd.Index):
        """Get loc references to train/test splits of ``y``.

        private _split containing the core logic, called from split_loc

        Parameters
        ----------
        y : pd.Index
            index of time series to split

        Yields
        ------
        train : pd.Index
            Training window indices, loc references to training indices in y
        test : pd.Index
            Test window indices, loc references to test indices in y
        """
        cv = self.cv

        for y_train_inner, y_test_inner in cv.split_loc(y):
            y_train_self = y_train_inner
            y_test_self = y_train_inner.union(y_test_inner)
            yield y_train_self, y_test_self

    def get_n_splits(self, y=None) -> int:
        """Return the number of splits.

        This will always be equal to the number of splits
        of ``self.cv`` on ``y``.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return self.cv.get_n_splits(y)

    def _fh(self):
        """Forecasting horizon, in integer resp array of integer, relative to cutoff.

        Private method called by property ``fh``,
        can be overridden by inheriting classes.

        Default is to return a forecasting horizon of ``1`` for temporal splitters,
        and ``None`` for instance splitters.

        If the attribute ``_fh_`` is set, then it is returned instead.

        Returns
        -------
        fh : array-like or int, optional, (default=None)
            Forecasting horizon with the steps ahead to predict, if splits are used
            for forecasting or backtesting.

            * if integer, the indices to forecast are ``1, 2, ..., fh``, periods ahead.
            * if array-like, the indices to forecast are given by the values in ``fh``,
              values must be coercible to integer.
            * ``None`` if no forecasting horizon is set. This is returned for splitters
              that do not have a natural forecasting horizon associated to them.
        """
        # inherits the fh from the cv splitter directly, since the splits are the same
        return self.cv.fh

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.split import ExpandingWindowSplitter, SingleWindowSplitter

        cv_1 = ExpandingWindowSplitter(fh=[2, 4], initial_window=5, step_length=2)
        cv_2 = SingleWindowSplitter(fh=[2, 4], window_length=3)
        return [{"cv": cv_1}, {"cv": cv_2}]
