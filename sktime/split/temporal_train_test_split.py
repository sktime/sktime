#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Single temporal train test split utility function."""

__all__ = [
    "temporal_train_test_split",
    "TemporalTrainTestSplitter",
]

import math
from typing import Optional

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import (
    ACCEPTED_Y_TYPES,
    FORECASTING_HORIZON_TYPES,
    SPLIT_TYPE,
    _split_by_fh,
)


def temporal_train_test_split(
    y: ACCEPTED_Y_TYPES,
    X: Optional[pd.DataFrame] = None,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    fh: Optional[FORECASTING_HORIZON_TYPES] = None,
    anchor: str = "start",
) -> SPLIT_TYPE:
    """Split time series data containers into a single train/test split.

    Creates a single train/test split of endogenous time series ``y``,
    an optionally exogeneous time series ``X``.

    Splits time series ``y`` into a single temporally ordered train and test split.
    The split is based on ``test_size`` and ``train_size`` parameters,
    which can signify fractions of total number of indices,
    or an absolute number of integers to cut.

    If the data contains multiple time series (Panel or Hierarchical),
    fractions and train-test sets will be computed per individual time series.

    If ``X`` is provided, will also produce a single train/test split of ``X``,
    at the same ``loc`` indices as ``y``. If non-``pandas`` based containers are used,
    will use ``iloc`` index instead.

    Parameters
    ----------
    y : time series in sktime compatible data container format
        endogenous time series
    X : time series in sktime compatible data container format, optional, default=None
        exogenous time series
    test_size : float, int or None, optional (default=None)
        If float, must be between 0.0 and 1.0, and is interpreted as the proportion
        of the dataset to include in the test split. Proportions are rounded to the
        next higher integer count of samples (ceil).
        If int, is interpreted as total number of test samples.
        If None, the value is set to the complement of the train size.
        If ``train_size`` is also None, it will be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, must be between 0.0 and 1.0, and is interpreted as the proportion
        of the dataset to include in the train split. Proportions are rounded to the
        next lower integer count of samples (floor).
        If int, is interpreted as total number of train samples.
        If None, the value is set to the complement of the test size.
    fh : ForecastingHorizon
        A forecast horizon to use for splitting, alternative specification for test set.
        If given, ``test_size`` and ``train_size`` cannot also be specified and must
        be None. If ``fh`` is passed, the test set will be:
        if ``fh.is_relative``: the last possible indices to match ``fh`` within ``y``
        if ``not fh.is_relative``: the indices at the absolute index of ``fh``
    anchor : str, "start" (default) or "end"
        determines behaviour if train and test sizes do not sum up to all data
        used only if ``fh=None`` and both ``test_size`` and ``train_size`` are not None
        if "start", cuts train and test set from start of available series
        if "end", cuts train and test set from end of available series

    Returns
    -------
    splitting : tuple, length = 2 * len(arrays)
        Tuple containing train-test split of ``y``, and ``X`` if given.
        if ``X is None``, returns ``(y_train, y_test)``.
        Else, returns ``(y_train, y_test, X_train, X_test)``.

    References
    ----------
    .. [1] originally adapted from https://github.com/alkaline-ml/pmdarima/

    Examples
    --------
    >>> from sktime.datasets import load_airline, load_osuleaf
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.utils._testing.panel import _make_panel
    >>> # univariate time series
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=36)
    >>> y_test.shape
    (36,)
    >>> # panel time series
    >>> y = _make_panel(n_instances = 2, n_timepoints = 20)
    >>> y_train, y_test = temporal_train_test_split(y, test_size=5)
    >>> # last 5 timepoints for each instance
    >>> y_test.shape
    (10, 1)

    The function can also be applied to panel or hierarchical data,
    in this case the split will be applied per individual time series:
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y = _make_hierarchical()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=0.2)
    """
    # the code has two disjoint branches, one for fh and one for test_size/train_size

    # branch 1: fh is not None, use fh to split
    # this assumes (or enforces) that test_size and train_size are None
    if fh is not None:
        if test_size is not None or train_size is not None:
            raise ValueError(
                "If `fh` is given, `test_size` and `train_size` cannot "
                "also be specified."
            )
        return _split_by_fh(y, fh, X=X)

    # branch 2: fh is None, use test_size and train_size to split
    # from the above, we know that fh is None
    temporal_splitter = TemporalTrainTestSplitter(
        test_size=test_size, train_size=train_size, anchor=anchor
    )

    y_train, y_test = list(temporal_splitter.split_series(y))[0]

    # if X is None, return y_train, y_test
    if X is None:
        return y_train, y_test

    # if X is not None, split X as well
    # the split of X uses the same indices as the split of y
    from sktime.split import SameLocSplitter

    X_splitter = SameLocSplitter(temporal_splitter, y)
    X_train, X_test = list(X_splitter.split_series(X))[0]

    return y_train, y_test, X_train, X_test


class TemporalTrainTestSplitter(BaseSplitter):
    r"""Temporal train-test splitter, based on sample sizes of train or test set.

    Cuts test and train sets from the start or end of available data,
    based on ``test_size`` and ``train_size`` parameters,
    which can signify fractions of total number of indices,
    or an absolute number of integers to cut.

    If the data contains multiple time series (Panel or Hierarchical),
    fractions and train-test sets will be computed per individual time series.

    Parameters
    ----------
    test_size : float, int or None, optional (default=None)
        If float, must be between 0.0 and 1.0, and is interpreted as the proportion
        of the dataset to include in the test split. Proportions are rounded to the
        next higher integer count of samples (ceil).
        If int, is interpreted as total number of test samples.
        If None, the value is set to the complement of the train size.
        If ``train_size`` is also None, it will be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, must be between 0.0 and 1.0, and is interpreted as the proportion
        of the dataset to include in the train split. Proportions are rounded to the
        next lower integer count of samples (floor).
        If int, is interpreted as total number of train samples.
        If None, the value is set to the complement of the test size.
    anchor : str, "start" (default) or "end"
        determines behaviour if train and test sizes do not sum up to all data
        if "start", cuts train and test set from start of available series
        if "end", cuts train and test set from end of available series

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import TemporalTrainTestSplitter
    >>> ts = np.arange(10)
    >>> splitter = TemporalTrainTestSplitter(test_size=0.3)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    """

    _tags = {"split_hierarchical": False}

    def __init__(self, train_size=None, test_size=None, anchor="start"):
        self.train_size = train_size
        self.test_size = test_size
        self.anchor = anchor
        super().__init__()

    def _split(self, y: pd.Index):
        test_size = self.test_size
        train_size = self.train_size
        anchor = self.anchor

        len_y = len(y)

        if test_size is None and train_size is None:
            test_size = 0.25

        if train_size is None:
            anchor = "end"
        if test_size is None:
            anchor = "start"

        if isinstance(test_size, float):
            test_size = math.ceil(test_size * len(y))
        if isinstance(train_size, float):
            train_size = math.floor(train_size * len(y))
        if test_size is None:
            test_size = len_y - train_size
        if train_size is None:
            train_size = len_y - test_size

        if anchor == "end":
            test_size = min(len_y, test_size)
            train_size = min(len_y - test_size, train_size)
        else:
            train_size = min(len_y, train_size)
            test_size = min(len_y - train_size, test_size)

        all_ix = np.arange(len_y)

        if anchor == "end":
            y_train_ix = all_ix[:-test_size]
            y_test_ix = all_ix[-test_size:]
            y_train_ix = y_train_ix[-train_size:]
        else:  # if anchor == "start"
            y_train_ix = all_ix[:train_size]
            y_test_ix = all_ix[train_size:]
            y_test_ix = y_test_ix[:test_size]

        yield y_train_ix, y_test_ix

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
        params1 = {"test_size": 0.2, "train_size": 0.3, "anchor": "start"}
        params2 = {"test_size": 0.2, "train_size": 0.3, "anchor": "end"}
        params3 = {"test_size": 2}
        params4 = {"train_size": 3}
        params5 = {}
        return [params1, params2, params3, params4, params5]
