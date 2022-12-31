#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement dataset splitting for model evaluation and selection."""

__all__ = [
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "CutoffSplitter",
    "SingleWindowSplitter",
    "temporal_train_test_split",
]
__author__ = ["mloning", "kkoralturk", "khrapovs", "chillerobscuro"]

from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split

from sktime.base import BaseObject
from sktime.datatypes import check_is_scitype, convert_to
from sktime.datatypes._utilities import get_index_for_series, get_time_index, get_window
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import VALID_FORECASTING_HORIZON_TYPES
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
    all_inputs_are_iloc_like,
    all_inputs_are_time_like,
    array_is_datetime64,
    array_is_int,
    check_window_length,
    is_datetime,
    is_int,
    is_timedelta,
    is_timedelta_or_date_offset,
)
from sktime.utils.validation.forecasting import (
    VALID_CUTOFF_TYPES,
    check_cutoffs,
    check_fh,
    check_step_length,
)
from sktime.utils.validation.series import check_equal_time_index

DEFAULT_STEP_LENGTH = 1
DEFAULT_WINDOW_LENGTH = 10
DEFAULT_FH = 1
ACCEPTED_Y_TYPES = Union[pd.Series, pd.DataFrame, np.ndarray, pd.Index]
FORECASTING_HORIZON_TYPES = Union[
    Union[VALID_FORECASTING_HORIZON_TYPES], ForecastingHorizon
]
SPLIT_TYPE = Union[
    Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]
]
SPLIT_ARRAY_TYPE = Tuple[np.ndarray, np.ndarray]
SPLIT_GENERATOR_TYPE = Iterator[SPLIT_ARRAY_TYPE]
PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _check_fh(fh: VALID_FORECASTING_HORIZON_TYPES) -> ForecastingHorizon:
    """Check and convert fh to format expected by CV splitters."""
    return check_fh(fh, enforce_relative=True)


def _get_end(y_index: pd.Index, fh: ForecastingHorizon) -> int:
    """Compute the end of the last training window for a forecasting horizon.

    For a time series index `y_index`, `y_index[end]` will give
    the index of the training window.
    Correspondingly, for a time series `y` with index `y_index`,
    `y.iloc[end]` or `y.loc[y_index[end]]`
    will provide the last index of the training window.

    Parameters
    ----------
    y_index : pd.Index
        Index of time series
    fh : int, timedelta, list or np.ndarray of ints or timedeltas

    Returns
    -------
    end : int
        0-indexed integer end of the training window
    """
    # `fh` is assumed to be ordered and checked by `_check_fh` and `window_length` by
    # `check_window_length`.
    n_timepoints = y_index.shape[0]
    assert isinstance(y_index, pd.Index)

    # For purely in-sample forecasting horizons, the last split point is the end of the
    # training data.
    # Otherwise, the last point must ensure that the last horizon is within the data.
    null = 0 if array_is_int(fh) else pd.Timedelta(0)
    fh_offset = null if fh.is_all_in_sample() else fh[-1]
    if array_is_int(fh):
        return n_timepoints - fh_offset - 1
    else:
        return y_index.get_loc(y_index[-1] - fh_offset)


def _check_window_lengths(
    y: pd.Index,
    fh: ForecastingHorizon,
    window_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
    initial_window: NON_FLOAT_WINDOW_LENGTH_TYPES,
) -> None:
    """Check that combination of inputs is compatible.

    Parameters
    ----------
    y : pd.Index
        Index of time series
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
    window_length : int or timedelta or pd.DateOffset
    initial_window : int or timedelta or pd.DateOffset
        Window length of first window

    Raises
    ------
    ValueError
        if window length plus max horizon is above the last observation in `y`,
        or if initial window plus max horizon is above the last observation in `y`
    TypeError
        if type of the input is not supported
    """
    n_timepoints = y.shape[0]
    fh_max = fh[-1]

    error_msg_for_incompatible_window_length = (
        f"The `window_length` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `window_length`={window_length}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the window length plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    if is_timedelta_or_date_offset(x=window_length):
        if y[0] + window_length + fh_max > y[-1]:
            raise ValueError(error_msg_for_incompatible_window_length)
    else:
        if window_length + fh_max > n_timepoints:
            raise ValueError(error_msg_for_incompatible_window_length)

    error_msg_for_incompatible_initial_window = (
        f"The `initial_window` and the forecasting horizon are incompatible "
        f"with the length of `y`. Found `initial_window`={initial_window}, "
        f"`max(fh)`={fh_max}, but len(y)={n_timepoints}. "
        f"It is required that the initial window plus maximum forecast horizon "
        f"is smaller than the length of the time series `y` itself."
    )
    error_msg_for_incompatible_types = (
        "The `initial_window` and `window_length` types are incompatible. "
        "They should be either all timedelta or all int."
    )
    if initial_window is not None:
        if is_timedelta_or_date_offset(x=initial_window):
            if y[0] + initial_window + fh_max > y[-1]:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if not is_timedelta_or_date_offset(x=window_length):
                raise TypeError(error_msg_for_incompatible_types)
        else:
            if initial_window + fh_max > n_timepoints:
                raise ValueError(error_msg_for_incompatible_initial_window)
            if is_timedelta_or_date_offset(x=window_length):
                raise TypeError(error_msg_for_incompatible_types)


def _inputs_are_supported(args: list) -> bool:
    """Check that combination of inputs is supported.

    Currently, only two cases are allowed:
    either all inputs are iloc-friendly, or they are all time-like

    Parameters
    ----------
    args : list of inputs to check

    Returns
    -------
    True if all inputs are compatible, False otherwise
    """
    if all_inputs_are_iloc_like(args) or all_inputs_are_time_like(args):
        return True
    else:
        return False


def _check_inputs_for_compatibility(args: list) -> None:
    """Check that combination of inputs is supported.

    Currently, only two cases are allowed:
    either all inputs are iloc-friendly, or they are time-like

    Parameters
    ----------
    args : list of inputs

    Raises
    ------
    TypeError
        if combination of inputs is not supported
    """
    if not _inputs_are_supported(args):
        raise TypeError("Unsupported combination of types")


def _check_cutoffs_and_y(cutoffs: VALID_CUTOFF_TYPES, y: ACCEPTED_Y_TYPES) -> None:
    """Check that combination of inputs is compatible.

    Parameters
    ----------
    cutoffs : np.array or pd.Index
        cutoff points, positive and integer- or datetime-index like
    y : pd.Series, pd.DataFrame, np.ndarray, or pd.Index
        coerced and checked version of input y

    Raises
    ------
    ValueError
        if max cutoff is above the last observation in `y`
    TypeError
        if `cutoffs` type is not supported
    """
    max_cutoff = np.max(cutoffs)
    msg = (
        "`cutoffs` are incompatible with given `y`. "
        "Maximum cutoff is not smaller than the "
    )
    if array_is_int(cutoffs):
        if max_cutoff >= y.shape[0]:
            raise ValueError(msg + "number of observations.")
    elif array_is_datetime64(cutoffs):
        if max_cutoff >= np.max(y):
            raise ValueError(msg + "maximum index value of `y`.")
    else:
        raise TypeError("Unsupported type of `cutoffs`")


def _check_cutoffs_fh_y(
    cutoffs: VALID_CUTOFF_TYPES, fh: FORECASTING_HORIZON_TYPES, y: pd.Index
) -> None:
    """Check that combination of inputs is compatible.

    Currently, only two cases are allowed:
    either both `cutoffs` and `fh` are integers, or they are datetime or timedelta.

    Parameters
    ----------
    cutoffs : np.array or pd.Index
        Cutoff points, positive and integer- or datetime-index like.
        Type should match the type of `fh` input.
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
        Type should match the type of `cutoffs` input.
    y : pd.Index
        Index of time series

    Raises
    ------
    ValueError
        if max cutoff plus max `fh` is above the last observation in `y`
    TypeError
        if `cutoffs` and `fh` type combination is not supported
    """
    max_cutoff = np.max(cutoffs)
    max_fh = fh.max()

    msg = "`fh` is incompatible with given `cutoffs` and `y`."
    if is_int(x=max_cutoff) and is_int(x=max_fh):
        if max_cutoff + max_fh > y.shape[0]:
            raise ValueError(msg)
    elif is_datetime(x=max_cutoff) and is_timedelta(x=max_fh):
        if max_cutoff + max_fh > y.max():
            raise ValueError(msg)
    else:
        raise TypeError("Unsupported type of `cutoffs` and `fh`")


class BaseSplitter(BaseObject):
    r"""Base class for temporal cross-validation splitters.

    The purpose of this implementation is to fill the gap relative to
    `sklearn.model_selection.TimeSeriesSplit
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>`__
    which implements only expanding window split strategy, and only integer based.

    The most important method in this class is `.split(y)` which generates indices
    of non-overlapping train/test splits of a time series `y`.
    The length of the train split is determined by `window_length`.
    The length of the test split is determined by forecasting horizon `fh`.

    In general, splitting a time series :math:`y=(y_1,\ldots,y_T)`
    into train/test splits means separating it into two non-overlapping series:
    train :math:`(y_{t(1)},\ldots,y_{t(k)})`
    and test :math:`(y_{t(k+1)},\ldots,y_{t(k+l)})`,
    where :math:`k,l` are all integers greater than zero,
    and :math:`t(k)<t(k+1)` are ordered time indices.
    The exact set of indices depends on a concrete splitter.
    Method `.split` is used to generate a pair of index sets:
    train :math:`(t(1),\ldots,t(k))` and test :math:`(t(k+1),\ldots,t(k+l))`.

    In case `window_length` and `fh` are integer valued,
    they translate into :math:`k` and :math:`l`, respectively.

    In case `window_length` and `fh` can be interpreted
    as time interval length (time deltas), then they correspond to
    :math:`t(k)-t(1)` and :math:`t(k+l)-t(k+1)`, respectively.

    Method `.get_n_splits` returns the number of splitting iterations.
    This number depends on a concrete splitting strategy and splitter parameters.

    Method `.get_cutoffs` returns the cutoff points between each train/test split.
    Using the above notation, for a single split it corresponds
    to the last integer index of the training window, :math:`k`

    In order to illustrate the difference in integer/interval arithmetic
    in calculating train/test indices, let us consider the following examples.
    Suppose, the arguments of a splitter are `cutoff = 10` and `window_length = 6`.
    Then, we have `train_start = cutoff - window_length = 4`.
    For timedelta-like values the logic is a bit more complicated.
    The time point corresponding to the `cutoff`
    (index value of the `y` series) is shifted back
    by the timedelta `window_length`,
    and then the integer position of the resulting datetime
    is considered to be the training window start.
    For example, for `cutoff = 10`, and `window_length = pd.Timedelta(6, unit="D")`,
    we have `y[cutoff] = pd.Timestamp("2021-01-10")`,
    and `y[cutoff] - window_length = pd.Timestamp("2021-01-04")`,
    which leads to `train_start = y.loc(y[cutoff] - window_length) = 4`.
    Similar timedelta arithmetic applies to other splitter arguments.

    Parameters
    ----------
    window_length : int or timedelta or pd.DateOffset
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
    ) -> None:
        self.window_length = window_length
        self.fh = fh

        super(BaseSplitter, self).__init__()

    def split(self, y: ACCEPTED_Y_TYPES) -> SPLIT_GENERATOR_TYPE:
        """Get iloc references to train/test slits of `y`.

        Parameters
        ----------
        y : pd.Index or time series in sktime compatible time series format,
                time series can be in any Series, Panel, or Hierarchical mtype format
            Index of time series to split, or time series to split
            If time series, considered as index of equivalent pandas type container:
                pd.DataFrame, pd.Series, pd-multiindex, or pd_multiindex_hier mtype

        Yields
        ------
        train : 1D np.ndarray of dtype int
            Training window indices, iloc references to training indices in y
        test : 1D np.ndarray of dtype int
            Test window indices, iloc references to test indices in y
        """
        y_index = self._coerce_to_index(y)

        if not isinstance(y_index, pd.MultiIndex):
            split = self._split
        else:
            split = self._split_vectorized

        for train, test in split(y_index):
            yield train[train >= 0], test[test >= 0]

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
        raise NotImplementedError("abstract method")

    def _split_vectorized(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        """Get iloc references to train/test splits of `y`, for pd.MultiIndex.

        This applies _split per time series instance in the multiindex.
        Instances in this context are defined by levels except last level.

        Parameters
        ----------
        y : pd.MultiIndex, with last level time-like
            as used in pd_multiindex and pd_multiindex_hier sktime mtypes

        Yields
        ------
        train : 1D np.ndarray of dtype int
            Training window indices, iloc references to training indices in y
        test : 1D np.ndarray of dtype int
            Test window indices, iloc references to test indices in y
        """
        # challenge is obtaining iloc references for *the original data frame*
        #   todo: try to shorten this, there must be a quicker way
        train_test_res = dict()
        train_iloc = dict()
        test_iloc = dict()
        y = pd.DataFrame(index=y)
        y_index_df = y.reset_index(-1)
        y_index_df["__index"] = range(len(y_index_df))
        y_index_inst = y_index_df.index.unique()
        for idx in y_index_inst:
            train_iloc[idx] = dict()
            test_iloc[idx] = dict()
        for idx in y_index_inst:
            y_inst = y_index_df.loc[idx]
            y_inst = y_inst.reset_index(drop=True).set_index(y_inst.columns[0])
            y_inst_index = y_inst.index
            train_test_res[idx] = list(self._split(y_inst_index))
            for i, tt in enumerate(train_test_res[idx]):
                train_iloc[idx][i] = tt[0]
                test_iloc[idx][i] = tt[1]
            for i, train in train_iloc[idx].items():
                train_iloc[idx][i] = y_inst["__index"].iloc[train].values
            for i, test in test_iloc[idx].items():
                test_iloc[idx][i] = y_inst["__index"].iloc[test].values

        train_multi = dict()
        test_multi = dict()
        for idx in y_index_inst:
            for i in train_iloc[idx].keys():
                if i not in train_multi.keys():
                    train_multi[i] = train_iloc[idx][i]
                    test_multi[i] = test_iloc[idx][i]
                else:
                    train_multi[i] = np.concatenate(
                        (train_iloc[idx][i], train_multi[i])
                    )
                    test_multi[i] = np.concatenate((test_iloc[idx][i], test_multi[i]))
        for i in train_multi.keys():
            train_multi[i] = np.sort(train_multi[i])
            test_multi[i] = np.sort(test_multi[i])

        for i in train_multi.keys():
            train = train_multi[i]
            test = test_multi[i]
            yield train, test

    def split_loc(self, y: ACCEPTED_Y_TYPES) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Get loc references to train/test splits of `y`.

        Parameters
        ----------
        y : pd.Index or time series in sktime compatible time series format,
                time series can be in any Series, Panel, or Hierarchical mtype format
            Time series to split, or index of time series to split

        Yields
        ------
        train : pd.Index
            Training window indices, loc references to training indices in y
        test : pd.Index
            Test window indices, loc references to test indices in y
        """
        y_index = self._coerce_to_index(y)

        for train, test in self.split(y_index):
            yield y_index[train], y_index[test]

    def split_series(self, y: ACCEPTED_Y_TYPES) -> Iterator[SPLIT_TYPE]:
        """Split `y` into training and test windows.

        Parameters
        ----------
        y : time series in sktime compatible time series format,
                time series can be in any Series, Panel, or Hierarchical mtype format
            e.g., pd.Series, pd.DataFrame, np.ndarray
            Time series to split, or index of time series to split

        Yields
        ------
        train : time series of same sktime mtype as `y`
            training series in the split
        test : time series of same sktime mtype as `y`
            test series in the split
        """
        y, y_orig_mtype = self._check_y(y)

        for train, test in self.split(y.index):
            y_train = y.iloc[train]
            y_test = y.iloc[test]
            y_train = convert_to(y_train, y_orig_mtype)
            y_test = convert_to(y_test, y_orig_mtype)
            yield y_train, y_test

    def _coerce_to_index(self, y: ACCEPTED_Y_TYPES) -> pd.Index:
        """Check and coerce y to pandas index.

        Parameters
        ----------
        y : pd.Index or time series in sktime compatible time series format (any)
            Index of time series to split, or time series to split
            If time series, considered as index of equivalent pandas type container:
                pd.DataFrame, pd.Series, pd-multiindex, or pd_multiindex_hier mtype

        Returns
        -------
        y_index : y, if y was pd.Index; otherwise _check_y(y).index
        """
        if not isinstance(y, pd.Index):
            y, _ = self._check_y(y, allow_index=True)
            y_index = y.index
        else:
            y_index = y
        return y_index

    def _check_y(self, y, allow_index=False):
        """Check and coerce y to a pandas based mtype.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D), optional (default=None)
            Time series to check, must conform with one of the sktime type conventions.

        Returns
        -------
        y_inner : time series y coerced to one of the sktime pandas based mtypes:
            pd.DataFrame, pd.Series, pd-multiindex, pd_multiindex_hier
            returns pd.Series only if y was pd.Series, otherwise a pandas.DataFrame
        y_mtype : original mtype of y

        Raises
        ------
        TypeError if y is not one of the permissible mtypes
        """
        if allow_index and isinstance(y, pd.Index):
            return y, "pd.Index"

        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]
        ALLOWED_MTYPES = [
            "pd.Series",
            "pd.DataFrame",
            "np.ndarray",
            "nested_univ",
            "numpy3D",
            # "numpyflat",
            "pd-multiindex",
            # "pd-wide",
            # "pd-long",
            "df-list",
            "pd_multiindex_hier",
        ]
        y_valid, _, y_metadata = check_is_scitype(
            y, scitype=ALLOWED_SCITYPES, return_metadata=True, var_name="y"
        )
        if allow_index:
            msg = (
                "y must be a pandas.Index, or a time series in an sktime compatible "
                "format, of scitype Series, Panel or Hierarchical, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and last(-1) level an sktime compatible time index."
                f" Allowed compatible mtype format specifications are: {ALLOWED_MTYPES}"
                "See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb, "
                "If you think y is already in an sktime supported input format, "
                "run sktime.datatypes.check_raise(y, mtype) to diagnose the error, "
                "where mtype is the string of the type specification you want for y. "
            )
        else:
            msg = (
                "y must be in an sktime compatible format, "
                "of scitype Series, Panel or Hierarchical, "
                "for instance a pandas.DataFrame with sktime compatible time indices, "
                "or with MultiIndex and last(-1) level an sktime compatible time index."
                f" Allowed compatible mtype format specifications are: {ALLOWED_MTYPES}"
                "See the forecasting tutorial examples/01_forecasting.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb, "
                "If you think y is already in an sktime supported input format, "
                "run sktime.datatypes.check_raise(y, mtype) to diagnose the error, "
                "where mtype is the string of the type specification you want for y. "
            )
        if not y_valid:
            raise TypeError(msg)

        y_inner = convert_to(y, to_type=PANDAS_MTYPES)

        mtype = y_metadata["mtype"]

        return y_inner, mtype

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
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

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        raise NotImplementedError("abstract method")

    def get_fh(self) -> ForecastingHorizon:
        """Return the forecasting horizon.

        Returns
        -------
        fh : ForecastingHorizon
            The forecasting horizon
        """
        return check_fh(self.fh)

    @staticmethod
    def _get_train_window(
        y: pd.Index, train_start: int, split_point: int
    ) -> np.ndarray:
        """Get train window.

        For formal definition of the train window see docstring of the `BaseSplitter`

        Parameters
        ----------
        y : pd.Index
            Index of time series to split
        train_start : int
            Integer index of the training window start
        split_point : int
            Integer index of the train window end

        Returns
        -------
        np.ndarray with integer indices of the train window

        """
        if split_point > max(0, train_start):
            return np.argwhere(
                (y >= y[max(train_start, 0)]) & (y <= y[min(split_point, len(y)) - 1])
            ).flatten()
        else:
            return np.array([], dtype=int)


class CutoffSplitter(BaseSplitter):
    r"""Cutoff window splitter.

    Split time series at given cutoff points into a fixed-length training and test set.

    Here the user is expected to provide a set of cutoffs (train set endpoints),
    which using the notation provided in :class:`BaseSplitter`,
    can be written as :math:`(k_1,\ldots,k_n)` for integer based indexing,
    or :math:`(t(k_1),\ldots,t(k_n))` for datetime based indexing.

    For a cutoff :math:`k_i` and a `window_length` :math:`w`
    the training window is :math:`(k_i-w+1,k_i-w+2,k_i-w+3,\ldots,k_i)`.
    Training window's last point is equal to the cutoff.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the test window will
    consist of the indices :math:`(k_n+h_1,\ldots, k_n+h_H)`.

    The number of splits returned by `.get_n_splits`
    is then trivially equal to :math:`n`.

    The sorted array of cutoffs returned by `.get_cutoffs` is then equal to
    :math:(t(k_1),\ldots,t(k_n))` with :math:`k_i<k_{i+1}`.

    Parameters
    ----------
    cutoffs : list or np.ndarray or pd.Index
        Cutoff points, positive and integer- or datetime-index like.
        Type should match the type of `fh` input.
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
        Type should match the type of `cutoffs` input.
    window_length : int or timedelta or pd.DateOffset

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.forecasting.model_selection import CutoffSplitter
    >>> ts = np.arange(10)
    >>> splitter = CutoffSplitter(fh=[2, 4], cutoffs=np.array([3, 5]), window_length=3)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([1, 2, 3]), array([5, 7])), (array([3, 4, 5]), array([7, 9]))]
    """

    def __init__(
        self,
        cutoffs: VALID_CUTOFF_TYPES,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
    ) -> None:
        _check_inputs_for_compatibility([fh, cutoffs, window_length])
        self.cutoffs = cutoffs
        super(CutoffSplitter, self).__init__(fh, window_length)

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        cutoffs = check_cutoffs(cutoffs=self.cutoffs)
        fh = _check_fh(fh=self.fh)
        window_length = check_window_length(
            window_length=self.window_length, n_timepoints=n_timepoints
        )
        if isinstance(y, (pd.DatetimeIndex, pd.PeriodIndex)) and is_int(window_length):
            window_length = y.freq * window_length
        _check_cutoffs_and_y(cutoffs=cutoffs, y=y)
        _check_cutoffs_fh_y(cutoffs=cutoffs, fh=fh, y=y)

        for cutoff in cutoffs:
            null = 0 if is_int(cutoff) else pd.Timestamp(0)
            if cutoff >= null:
                train_end = y[cutoff] if is_int(cutoff) else cutoff
                training_window = get_window(
                    pd.Series(index=y[y <= train_end]), window_length=window_length
                ).index
            else:
                training_window = []
            training_window = y.get_indexer(training_window)
            test_window = cutoff + fh.to_numpy()
            if is_datetime(x=cutoff):
                test_window = y.get_indexer(test_window[test_window >= y.min()])
            yield training_window, test_window

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        For this splitter the number is trivially equal to
        the number of cutoffs given during instance initialization.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return len(self.cutoffs)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        This method trivially returns the cutoffs given during instance initialization,
        in case these cutoffs are integer .iloc[] friendly indices.
        The only change is that the set of cutoffs is sorted from smallest to largest.
        When the given cutoffs are datetime-like,
        then this method returns corresponding integer indices.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if array_is_int(self.cutoffs):
            return check_cutoffs(self.cutoffs)
        else:
            return np.argwhere(y.index.isin(check_cutoffs(self.cutoffs))).flatten()

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
        params = {"cutoffs": np.array([3, 7, 10])}
        return params


class BaseWindowSplitter(BaseSplitter):
    """Base class for sliding and expanding window splitter."""

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES,
        start_with_window: bool,
    ) -> None:
        _check_inputs_for_compatibility(
            [fh, initial_window, window_length, step_length]
        )
        self.step_length = step_length
        self.start_with_window = start_with_window
        self.initial_window = initial_window
        super(BaseWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    @property
    def _initial_window(self):
        if hasattr(self, "initial_window"):
            return self.initial_window
        else:
            return None

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        window_length = check_window_length(
            window_length=self.window_length,
            n_timepoints=n_timepoints,
            name="window_length",
        )
        initial_window = check_window_length(
            window_length=self._initial_window,
            n_timepoints=n_timepoints,
            name="initial_window",
        )
        fh = _check_fh(self.fh)
        _check_window_lengths(
            y=y, fh=fh, window_length=window_length, initial_window=initial_window
        )

        if self._initial_window is not None:
            yield self._split_for_initial_window(y)

        for train, test in self._split_windows(window_length=window_length, y=y, fh=fh):
            yield train, test

    def _split_for_initial_window(self, y: pd.Index) -> SPLIT_ARRAY_TYPE:
        """Get train/test splits for non-empty initial window.

        Parameters
        ----------
        y : pd.Index
            Index of the time series to split

        Returns
        -------
        (np.ndarray, np.ndarray)
            Integer indices of the train/test windows

        """
        fh = _check_fh(self.fh)
        if not self.start_with_window:
            raise ValueError(
                "`start_with_window` must be True if `initial_window` is given"
            )
        if self._initial_window <= self.window_length:
            raise ValueError("`initial_window` must greater than `window_length`")
        if is_int(x=self._initial_window):
            end = self._initial_window
        else:
            end = y.get_loc(y[0] + self._initial_window)
        train = self._get_train_window(y=y, train_start=0, split_point=end)
        if array_is_int(fh):
            test = end + fh.to_numpy() - 1
        else:
            test = np.argwhere(y.isin(y[end - 1] + fh)).flatten()
        return train, test

    def _split_windows(
        self,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: pd.Index,
        fh: ForecastingHorizon,
    ) -> SPLIT_GENERATOR_TYPE:
        """Abstract method for sliding/expanding windows."""
        raise NotImplementedError("abstract method")

    def _split_windows_generic(
        self,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
        y: pd.Index,
        fh: ForecastingHorizon,
        expanding: bool,
    ) -> SPLIT_GENERATOR_TYPE:
        """Split `y` into training and test windows.

        This function encapsulates common functionality
        shared by concrete implementations of this abstract class.

        Parameters
        ----------
        window_length : int or timedelta or pd.DateOffset
            Length of training window
        y : pd.Index
            Index of time series to split
        fh : ForecastingHorizon
            Single step ahead or array of steps ahead to forecast.
        expanding : bool
            Expanding (True) or sliding window (False) splitter

        Yields
        ------
        train : 1D np.ndarray of int
            Training window iloc indices, in reference to y
        test : 1D np.ndarray of int
            Test window iloc indices, in reference to y
        """
        start = self._get_start(y=y, fh=fh)
        split_points = self.get_cutoffs(pd.Series(index=y, dtype=float)) + 1
        split_points = (
            split_points if self._initial_window is None else split_points[1:]
        )
        for split_point in split_points:
            train_start = self._get_train_start(
                start=start if expanding else split_point,
                window_length=window_length,
                y=y,
            )
            train = self._get_train_window(
                y=y, train_start=train_start, split_point=split_point
            )
            if array_is_int(fh):
                test = split_point + fh.to_numpy() - 1
            else:
                test = np.argwhere(
                    y.isin(y[max(0, split_point - 1)] + fh.to_pandas())
                ).flatten()
                if split_point == 0:
                    test -= 1
            yield train, test

    @staticmethod
    def _get_train_start(
        start: int, window_length: ACCEPTED_WINDOW_LENGTH_TYPES, y: pd.Index
    ) -> int:
        if is_timedelta_or_date_offset(x=window_length):
            train_start = y.get_loc(
                max(y[min(start, len(y) - 1)] - window_length, min(y))
            )
            if start >= len(y):
                train_start += 1
        else:
            train_start = start - window_length
        return train_start

    def _get_start(self, y: pd.Index, fh: ForecastingHorizon) -> int:
        """Get the first split point."""
        # By default, the first split point is the index zero, the first
        # observation in
        # the data.
        start = 0

        # If we start with a full window, the first split point depends on the window
        # length.
        if hasattr(self, "start_with_window") and self.start_with_window:

            if self._initial_window not in [None, 0]:

                if is_timedelta_or_date_offset(x=self._initial_window):
                    start = y.get_loc(
                        y[start] + self._initial_window + self.step_length
                    )
                else:
                    start += self._initial_window + self.step_length
            else:
                if is_timedelta_or_date_offset(x=self.window_length):
                    start = y.get_loc(y[start] + self.window_length)
                else:
                    start += self.window_length

        # For in-sample forecasting horizons, the first split must ensure that
        # in-sample test set is still within the data.
        if not fh.is_all_out_of_sample():
            fh_min = abs(fh[0])
            if is_int(fh_min):
                start = fh_min + 1 if fh_min >= start else start
            else:
                shifted_y0 = y[0] + fh_min
                start = np.argmin(y <= shifted_y0) if shifted_y0 >= y[start] else start
        return start

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
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
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the "
                f"number of splits."
            )
        return len(self.get_cutoffs(y))

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        y = get_index_for_series(y)
        fh = _check_fh(self.fh)
        step_length = check_step_length(self.step_length)

        if self._initial_window is None:
            start = self._get_start(y=y, fh=fh)
        elif is_int(x=self._initial_window):
            start = self._initial_window
        else:
            start = y.get_loc(y[0] + self._initial_window)

        end = _get_end(y_index=y, fh=fh) + 2
        if is_int(x=step_length):
            return np.arange(start, end, step_length) - 1
        else:
            offset = step_length if start == 0 else pd.Timedelta(0)
            start_date = y[y < y[start] + offset][-1]
            end_date = y[end - 1] - step_length if end <= len(y) else y[-1]
            date_cutoffs = pd.date_range(
                start=start_date, end=end_date, freq=step_length
            )
            cutoffs = np.argwhere(y.isin(date_cutoffs)).flatten()
            if start <= 0:
                cutoffs = np.hstack((-1, cutoffs))
            return cutoffs


class SlidingWindowSplitter(BaseWindowSplitter):
    r"""Sliding window splitter.

    Split time series repeatedly into a fixed-length training and test set.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the training window will
    consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`.

    For example for `window_length = 5`, `step_length = 1` and `fh = [1, 2, 3]`
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
    window_length : int or timedelta or pd.DateOffset
        Window length
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    initial_window : int or timedelta or pd.DateOffset, optional (default=None)
        Window length of first window
    start_with_window : bool, optional (default=True)
        - If True, starts with full window.
        - If False, starts with empty window.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.forecasting.model_selection import SlidingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = SlidingWindowSplitter(fh=[2, 4], window_length=3, step_length=2)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([0, 1, 2]), array([4, 6])), (array([2, 3, 4]), array([6, 8]))]

    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
        start_with_window: bool = True,
    ) -> None:
        super(SlidingWindowSplitter, self).__init__(
            fh=fh,
            window_length=window_length,
            initial_window=initial_window,
            step_length=step_length,
            start_with_window=start_with_window,
        )

    def _split_windows(self, **kwargs) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(expanding=False, **kwargs)


class ExpandingWindowSplitter(BaseWindowSplitter):
    r"""Expanding window splitter.

    Split time series repeatedly into an growing training set and a fixed-size test set.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the training window will
    consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`.

    For example for `initial_window = 5`, `step_length = 1` and `fh = [1, 2, 3]`
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
    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Window length of initial training fold. If =0, initial training fold is empty.
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = ExpandingWindowSplitter(fh=[2, 4], initial_window=5, step_length=2)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    '[(array([0, 1, 2, 3, 4]), array([6, 8]))]'

    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
    ) -> None:

        start_with_window = initial_window != 0

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

        # initial_window needs to be written to self for sklearn compatibility
        self.initial_window = initial_window
        # this class still acts as if it were overwritten with None,
        # via the _initial_window property that is read everywhere

    @property
    def _initial_window(self):
        return None

    def _split_windows(self, **kwargs) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(expanding=True, **kwargs)


class SingleWindowSplitter(BaseSplitter):
    r"""Single window splitter.

    Split time series once into a training and test set.
    See more details on what to expect from this splitter in :class:`BaseSplitter`.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the training window will
    consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int or timedelta or pd.DateOffset
        Window length

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.forecasting.model_selection import SingleWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = SingleWindowSplitter(fh=[2, 4], window_length=3)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([3, 4, 5]), array([7, 9]))]

    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        window_length: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
    ) -> None:
        _check_inputs_for_compatibility(args=[fh, window_length])
        super(SingleWindowSplitter, self).__init__(fh=fh, window_length=window_length)

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        window_length = check_window_length(self.window_length, n_timepoints)
        if isinstance(y, (pd.DatetimeIndex, pd.PeriodIndex)) and is_int(window_length):
            window_length = y.freq * window_length
        fh = _check_fh(self.fh)
        train_end = _get_end(y_index=y, fh=fh)

        training_window = get_window(
            pd.Series(index=y[y <= y[train_end]]), window_length=window_length
        ).index
        training_window = y.get_indexer(training_window)
        if array_is_int(fh):
            test_window = train_end + fh.to_numpy()
        else:
            test_window = y.get_indexer(y[train_end] + fh.to_pandas())

        yield training_window, test_window

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

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Since this splitter returns a single train/test split,
        this method returns a single one-dimensional array
        with the last train set index.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        fh = _check_fh(self.fh)
        y = get_index_for_series(y)
        end = _get_end(y_index=y, fh=fh)
        return np.array([end])

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
        params = {"fh": 3}
        return params


def temporal_train_test_split(
    y: ACCEPTED_Y_TYPES,
    X: Optional[pd.DataFrame] = None,
    test_size: Optional[Union[int, float]] = None,
    train_size: Optional[Union[int, float]] = None,
    fh: Optional[FORECASTING_HORIZON_TYPES] = None,
) -> SPLIT_TYPE:
    """Split arrays or matrices into sequential train and test subsets.

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
    splitting : tuple, length=2 * len(arrays)
        List containing train-test split of `y` and `X` if given.

    References
    ----------
    .. [1]  adapted from https://github.com/alkaline-ml/pmdarima/
    """
    if fh is not None:
        if test_size is not None or train_size is not None:
            raise ValueError(
                "If `fh` is given, `test_size` and `train_size` cannot "
                "also be specified."
            )
        return _split_by_fh(y, fh, X=X)
    else:
        pd_format = isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
        if pd_format is True and isinstance(y.index, pd.MultiIndex):
            ys = get_time_index(y)
            # Get index to group across (only indices other than timepoints index)
            yi_name = y.index.names
            yi_grp = yi_name[0:-1]

            # Get split into test and train data for timeindex only
            series = (ys,)
            yret = _train_test_split(
                *series,
                shuffle=False,
                stratify=None,
                test_size=test_size,
                train_size=train_size,
            )

            # Convert into list indices
            ysl = ys.to_list()
            yrl1 = yret[0].to_list()
            yrl2 = yret[1].to_list()
            p1 = [index for (index, item) in enumerate(ysl) if item in yrl1]
            p2 = [index for (index, item) in enumerate(ysl) if item in yrl2]

            # Subset by group based on identified indices
            y_train = y.groupby(yi_grp, as_index=False).nth(p1)
            y_test = y.groupby(yi_grp, as_index=False).nth(p2)
            if X is not None:
                X_train = X.groupby(yi_grp, as_index=False).nth(p1)
                X_test = X.groupby(yi_grp, as_index=False).nth(p2)
                return y_train, y_test, X_train, X_test
            else:
                return y_train, y_test
        else:
            series = (y,) if X is None else (y, X)
            return _train_test_split(
                *series,
                shuffle=False,
                stratify=None,
                test_size=test_size,
                train_size=train_size,
            )


def _split_by_fh(
    y: ACCEPTED_Y_TYPES, fh: FORECASTING_HORIZON_TYPES, X: Optional[pd.DataFrame] = None
) -> SPLIT_TYPE:
    """Split time series with forecasting horizon.

    Handles both relative and absolute horizons.
    """
    if X is not None:
        check_equal_time_index(y, X)
    index = y.index
    fh = check_fh(fh, freq=index)
    idx = fh.to_pandas()

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
