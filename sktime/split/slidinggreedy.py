#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that combines sliding window with greedy splitting."""

__author__ = ["marrov"]

__all__ = [
    "SlidingGreedySplitter",
]


import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter


class SlidingGreedySplitter(BaseSplitter):
    """Splitter that combines sliding window with greedy splitting.

    Takes integers or floats ``train_size`` and ``test_size`` that define the
    number of steps included in the train and test sets of each fold.
    If float values are provided, they are interpreted as proportions of the
    total length of the time series. If the data contains multiple instances,
    sizes are _per instance_.

    If no ``step_length`` is defined, the test sets (one for each fold) will be
    adjacent and disjoint, taken from the end of the dataset.

    Parameters
    ----------
    train_size : int or float
        If int: the number of steps included in the train set of each fold.
            Formally, steps are consecutive ``iloc`` indices.
        If float: the proportion of steps included in the train set of each fold,
            as a proportion of the total number of consecutive ``iloc`` indices.
            Must be between 0.0 and 1.0. Proportions are rounded to the
            next higher integer count of samples (ceil).
    test_size : int or float
        If int: the number of steps included in the test set of each fold.
            Formally, steps are consecutive ``iloc`` indices.
        If float: the proportion of steps included in the test set of each fold,
            as a proportion of the total number of consecutive ``iloc`` indices.
            Must be between 0.0 and 1.0. Proportions are rounded to the
            next higher integer count of samples (ceil).
    folds : int, default = 5
        The number of folds.
    step_length : int, optional
        The number of steps advanced for each fold. Defaults to ``test_size``.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import SlidingGreedySplitter

    >>> ts = np.arange(10)
    >>> splitter = SlidingGreedySplitter(train_size=4, test_size=2, folds=2)
    >>> list(splitter.split(ts))  # doctest: +SKIP
    [
        (array([2, 3, 4, 5]), array([6, 7])),
        (array([4, 5, 6, 7]), array([8, 9]))
    ]
    """

    _tags = {"split_hierarchical": True}

    def __init__(
        self,
        train_size: int | float,
        test_size: int | float,
        folds: int = 5,
        step_length: int | None = None,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.folds = folds
        self.step_length = step_length
        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if isinstance(self.train_size, float) or isinstance(self.test_size, float):
            self.set_tags(**{"split_hierarchical": False})

    def _split(self, y: pd.Index):
        train_size = self.train_size
        test_size = self.test_size

        if isinstance(train_size, float):
            _train_size = int(np.ceil(len(y) * train_size))
        else:
            _train_size = train_size

        if isinstance(test_size, float):
            _test_size = int(np.ceil(len(y) * test_size))
            self.fh = np.arange(_test_size) + 1
        else:
            _test_size = test_size

        if isinstance(y, pd.MultiIndex):
            groups = pd.Series(index=y, dtype="float64").groupby(y.names[:-1])
            reverse_idx = groups.transform("size") - groups.cumcount() - 1
        else:
            reverse_idx = np.arange(len(y))[::-1]

        step_length = self.step_length or _test_size

        for i in reversed(range(self.folds)):
            tst_end = i * step_length
            tst_start = tst_end + _test_size
            trn_end = tst_start
            trn_start = trn_end + _train_size

            trn_indices = np.flatnonzero(
                (reverse_idx >= trn_end) & (reverse_idx < trn_start)
            )
            tst_indices = np.flatnonzero(
                (reverse_idx >= tst_end) & (reverse_idx < tst_start)
            )

            if len(trn_indices) > 0 and len(tst_indices) > 0:
                yield trn_indices, tst_indices

    def _fh(self):
        """Forecasting horizon, in integer resp array of integer, relative to cutoff.

        Private method called by property ``fh``,
        can be overridden by inheriting classes.

        Default is to return a forecasting horizon of ``1``.

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
        test_size = self.test_size
        fh = np.arange(test_size) + 1 if isinstance(test_size, int) else None
        # if test_size is float, fh is set later in _split, so we check for that here
        if fh is None and hasattr(self, "_fh_"):
            fh = self._fh_
        return fh

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
        """
        params1 = {"train_size": 3, "test_size": 1}
        params2 = {"train_size": 4, "test_size": 2, "folds": 2, "step_length": 2}
        params3 = {"train_size": 0.3, "test_size": 0.2, "folds": 2}
        return [params1, params2, params3]
