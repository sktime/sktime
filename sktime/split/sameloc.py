#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that replicates loc indices from another splitter."""

__author__ = ["fkiraly"]

__all__ = [
    "SameLocSplitter",
]

from typing import Optional

import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import ACCEPTED_Y_TYPES, SPLIT_GENERATOR_TYPE


class SameLocSplitter(BaseSplitter):
    r"""Splitter that replicates loc indices from another splitter.

    Takes a splitter ``cv`` and a time series ``y_template``.
    Splits ``y`` in ``split`` and ``split_loc`` such that ``loc`` indices of splits
    are identical to loc indices of ``cv`` applied to ``y_template``.

    Parameters
    ----------
    cv : BaseSplitter
        splitter for which to replicate splits by ``loc`` index
    y_template : time series container of ``Series`` scitype, optional
        template used in ``cv`` to determine ``loc`` indices
        if None, ``y_template=y`` will be used in methods

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.split import (
    ...    ExpandingWindowSplitter,
    ...    SameLocSplitter,
    ... )

    >>> y = load_airline()
    >>> y_template = y[:60]
    >>> cv_tpl = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)

    >>> splitter = SameLocSplitter(cv_tpl, y_template)

    these two are the same:

    >>> list(cv_tpl.split(y_template)) # doctest: +SKIP
    >>> list(splitter.split(y)) # doctest: +SKIP
    """

    _tags = {
        "split_hierarchical": True,
        # SameLocSplitter supports hierarchical pandas index
        "split_series_uses": "loc",
        # loc is quicker to get since that is directly passed
    }

    def __init__(self, cv, y_template=None):
        self.cv = cv
        self.y_template = y_template
        super().__init__()

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        cv = self.cv
        if self.y_template is None:
            y_template = y
        else:
            y_template = self.y_template

        for y_train_loc, y_test_loc in cv.split_loc(y_template):
            y_train_iloc = y.get_indexer(y_train_loc)
            y_test_iloc = y.get_indexer(y_test_loc)
            yield y_train_iloc, y_test_iloc

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
        if self.y_template is None:
            y_template = y
        else:
            y_template = self.y_template

        yield from cv.split_loc(y_template)

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        This will always be equal to the number of splits
        of ``self.cv`` on ``self.y_template``.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if self.y_template is None:
            y_template = y
        else:
            y_template = self.y_template
        return self.cv.get_n_splits(y_template)

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
        from sktime.datasets import load_airline
        from sktime.forecasting.model_selection import (
            ExpandingWindowSplitter,
            SingleWindowSplitter,
        )

        y = load_airline()
        y_temp = y[:60]
        cv_1 = ExpandingWindowSplitter(fh=[2, 4], initial_window=24, step_length=12)
        cv_2 = SingleWindowSplitter(fh=[2, 4], window_length=24)
        return [{"cv": cv_1, "y_template": y_temp}, {"cv": cv_2, "y_template": y_temp}]
