#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that successively cuts test folds off the end of the series."""

__author__ = ["davidgilbertson"]

__all__ = [
    "ExpandingGreedySplitter",
]
__author__ = ["davidgilbertson"]

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import SPLIT_GENERATOR_TYPE


class ExpandingGreedySplitter(BaseSplitter):
    """Splitter that successively cuts test folds off the end of the series.

    Takes an integer ``test_size`` that defines the number of steps included in the
    test set of each fold. The train set of each fold will contain all data before
    the test set. If the data contains multiple instances, ``test_size`` is
    _per instance_.

    If no ``step_length`` is defined, the test sets (one for each fold) will be
    adjacent and disjoint, taken from the end of the dataset.

    For example, with ``test_size=7`` and ``folds=5``, the test sets in total will cover
    the last 35 steps of the data with no overlap.

    Parameters
    ----------
    test_size : int or float
        If int: the number of steps included in the test set of each fold.
            Formally, steps are consecutive ``iloc`` indices.
        If float: the proportion of steps included in the test set of each fold,
            as a proportion of the total number of consecutive ``iloc`` indices.
            Must be between 0.0 and 1.0. Proportions are rounded to the
            next higher integer count of samples (ceil).
            Cave: not the ``loc`` proportion between start and end locations,
            but a proportion of total number of consecutive ``iloc`` indices.
    folds : int, default = 5
        The number of folds.
    step_length : int, optional
        The number of steps advanced for each fold. Defaults to ``test_size``.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ExpandingGreedySplitter

    >>> ts = np.arange(10)
    >>> splitter = ExpandingGreedySplitter(test_size=3, folds=2)
    >>> list(splitter.split(ts))  # doctest: +SKIP
    [
        (array([0, 1, 2, 3]), array([4, 5, 6])),
        (array([0, 1, 2, 3, 4, 5, 6]), array([7, 8, 9]))
    ]
    """

    _tags = {"split_hierarchical": True}

    def __init__(self, test_size: int, folds: int = 5, step_length: int = None):
        super().__init__()
        self.test_size = test_size
        self.folds = folds
        self.step_length = step_length
        self.fh = np.arange(test_size) + 1

        # no algorithm implemented that is faster for float than naive iteration
        if isinstance(test_size, float):
            self.set_tags(**{"split_hierarchical": False})

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        test_size = self.test_size

        if isinstance(test_size, float):
            _test_size = np.ceil(len(y) * test_size)
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
            trn_end = tst_end + _test_size
            trn_indices = np.flatnonzero(reverse_idx >= trn_end)
            tst_indices = np.flatnonzero(
                (reverse_idx < trn_end) & (reverse_idx >= tst_end)
            )
            yield trn_indices, tst_indices

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
        params1 = {"test_size": 1}
        params2 = {"test_size": 3, "folds": 2, "step_length": 2}
        params3 = {"test_size": 0.2, "folds": 2}
        return [params1, params2, params3]
