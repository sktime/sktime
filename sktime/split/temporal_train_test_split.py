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
    """Create a single temporal train-test split from time series data.

    The function splits time series data into one training and one test set.

    For a time series with time points (t₁, t₂, ..., tₙ), the function creates:
    1. A training set containing indices in [t₁, t_k)
    2. A test set containing indices in [t_k, tₙ)

    where t_k is determined by either:
    * train_size/test_size parameters
    * forecasting horizon fh
    * anchor position ("start" or "end")

    The split can be created in two ways:

    1. Using train_size/test_size:
       - If anchor="start": training set starts at t₁, test set follows
       - If anchor="end": test set ends at tₙ, training set precedes
       - Any remaining indices are excluded from both sets

    2. Using forecasting horizon fh:
       - For relative horizons: test set contains the last indices that match fh
       - For absolute horizons: test set contains indices at fh's absolute positions

    If data contains multiple series (Panel/Hierarchical), splits are computed 
    per individual time series.

    Parameters
    ----------
    y : time series in sktime compatible data container format
        Endogenous time series to split
    X : time series in sktime compatible data container format, optional (default=None)
        Optional exogenous time series to split at same indices as y
    test_size : float, int or None, optional (default=None)
        Size of the test set
        - float: proportion of total samples (0.0 to 1.0), rounded up
        - int: absolute number of test samples
        - None: computed as complement of train_size
                If train_size also None, defaults to 0.25
    train_size : float, int, or None, optional (default=None)
        Size of the training set
        - float: proportion of total samples (0.0 to 1.0), rounded down
        - int: absolute number of training samples  
        - None: computed as complement of test_size
    fh : ForecastingHorizon, optional (default=None)
        Alternative way to specify test set using forecast horizon
        - Cannot be combined with test_size or train_size
        - Relative fh: test set contains last possible matching indices
        - Absolute fh: test set contains indices at fh positions
    anchor : str, {"start", "end"}, optional (default="start")
        When train/test sizes don't use all data:
        - "start": cut from beginning of series
        - "end": cut from end of series
        Only used if fh=None and both sizes specified

    Returns
    -------
    splitting : tuple, length = 2 * len(arrays)
        Train-test splits:
        - Without X: (y_train, y_test)
        - With X: (y_train, y_test, X_train, X_test)

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

    Creates a single train-test split by partitioning time series data into
    training and test sets based on sample size parameters.

    For a time series with time points (t₁, t₂, ..., tₙ), creates:
    1. A training set with indices [t_start, t_start + train_size)
    2. A test set with indices [t_test, t_test + test_size)

    where t_start and t_test depend on the anchor parameter:
    - If anchor="start": t_start = t₁, t_test = t₁ + train_size
    - If anchor="end": t_test = tₙ - test_size, t_start = t_test - train_size

    For panel/hierarchical data, split is computed per individual time series.

    Parameters
    ----------
    test_size : float, int or None, optional (default=None)
        Size of the test set
        - float: proportion of total samples (0.0 to 1.0), rounded up
        - int: absolute number of test samples
        - None: computed as complement of train_size
                If train_size also None, defaults to 0.25
    train_size : float, int, or None, optional (default=None)
        Size of the training set
        - float: proportion of total samples (0.0 to 1.0), rounded down 
        - int: absolute number of training samples
        - None: computed as complement of test_size
    anchor : str, {"start", "end"}, optional (default="start")
        Determines how to position train/test windows:
        - "start": cut from beginning of series 
        - "end": cut from end of series

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