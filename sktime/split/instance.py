#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that uses an sklearn splitter to split by instance."""

__author__ = ["fkiraly", "ksharma6"]

__all__ = [
    "InstanceSplitter",
]

from typing import Optional

import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import ACCEPTED_Y_TYPES, SPLIT_GENERATOR_TYPE
from sktime.utils.multiindex import apply_split


class InstanceSplitter(BaseSplitter):
    r"""Splitter that applies an sklearn instance splitter to a panel of time series.

    For splitter ``cv``, applies ``cv`` to the instance index of ``y``.
    The instance index are all levels of a hierarchical or panel index,
    except the last, temporal one.

    Returned split iloc and loc indices are in reference to the original ``y``,
    not the instance index.

    Parameters
    ----------
    cv : sklearn splitter
        splitter for the instance index

    Examples
    --------
    >>> from sktime.split import InstanceSplitter
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sklearn.model_selection import KFold
    >>>
    >>> y = _make_hierarchical()
    >>> splitter = InstanceSplitter(KFold(n_splits=3))
    >>> list(splitter.split(y))  # doctest: +SKIP
    """

    _tags = {
        "split_hierarchical": True,
        # SameLocSplitter supports hierarchical pandas index
        "split_series_uses": "iloc",
        # iloc is implemented directly
        "split_type": "instance",
    }

    def __init__(self, cv):
        self.cv = cv
        super().__init__()

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        """Generate indices to split data."""
        if not isinstance(y, pd.MultiIndex):
            zeros = [0] * len(y)
            y = pd.MultiIndex.from_arrays([zeros, y])

        inst_ix = y.droplevel(-1).unique()

        for y_train_inst_iloc, y_test_inst_iloc in self.cv.split(inst_ix):
            y_train_iloc = apply_split(y, y_train_inst_iloc)
            y_test_iloc = apply_split(y, y_test_inst_iloc)
            yield y_train_iloc, y_test_iloc

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
        if not isinstance(y, pd.Index):
            y = self._check_y(y, allow_index=True)[0]
            y = y.index
        inst_ix = y.droplevel(-1).unique()
        return self.cv.get_n_splits(inst_ix)

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
        from sklearn.model_selection import KFold, ShuffleSplit

        params1 = {"cv": KFold(n_splits=3)}
        params2 = {"cv": ShuffleSplit(n_splits=3)}
        return [params1, params2]
