#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that modifies a splitter by adding train folds to the test folds."""

__author__ = ["fkiraly"]

__all__ = ["TestPlusTrainSplitter"]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import ACCEPTED_Y_TYPES, SPLIT_GENERATOR_TYPE


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
    >>> from sktime.split import ExpandingWindowSplitter

    >>> y = load_airline()
    >>> y_template = y[:60]
    >>> cv_tpl = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)

    >>> splitter = TestPlusTrainSplitter(cv_tpl)
    """

    def __init__(self, cv):
        self.cv = cv
        super().__init__()

        # dispatch split_series to the same split/split_loc as the wrapped cv
        # for performance reasons
        self.clone_tags(cv, "split_series_uses")

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        """Get iloc references to train/test splits of `y`.

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

    def _split_loc(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        """Get loc references to train/test splits of `y`.

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

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.model_selection import (
            ExpandingWindowSplitter,
            SingleWindowSplitter,
        )

        cv_1 = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)
        cv_2 = SingleWindowSplitter(fh=[2, 4], window_length=24)
        return [{"cv": cv_1}, {"cv": cv_2}]
