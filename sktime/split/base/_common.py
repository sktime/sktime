#!/usr/bin/env python3 -u
"""Common utilities and constants for time series splitter module."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sktime.datatypes._utilities import get_window
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import VALID_FORECASTING_HORIZON_TYPES
from sktime.utils.validation import (
    all_inputs_are_iloc_like,
    all_inputs_are_time_like,
    array_is_int,
    is_int,
)
from sktime.utils.validation.forecasting import check_fh
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
    return all_inputs_are_iloc_like(args) or all_inputs_are_time_like(args)


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
    return y_index.get_loc(y_index[-1] - fh_offset)


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

    X_train = X.loc[train]
    X_test = X.loc[test]
    return y_train, y_test, X_train, X_test


def _get_train_window_via_endpoint(y, train_endpoint, window_length):
    """
    Split time series at given end points into a fixed-length training set.

    Parameters
    ----------
    y : pd.Index
        Index of time series to split
    train_endpoint : int or timedelta
        Training window's last time point
    window_length : int or timedelta
        Length of window

    Returns
    -------
    training_window : pd.Index
        Training window indices

    Notes
    -----
    this private function is used to get training window for
    `CutOffSplitter` and `SingleWindowSplitter`
    """
    if isinstance(y, (pd.DatetimeIndex, pd.PeriodIndex)) and is_int(window_length):
        y_train = pd.Series(index=y, dtype=y.dtype)  # convert pd.index to pd.series
        train_start = train_endpoint - window_length + 1
        # adjust start point to account for negative time point
        train_start = 0 if train_start < 0 else train_start
        train_window = y_train.iloc[train_start : train_endpoint + 1].index
    else:
        train_end = y[train_endpoint] if is_int(train_endpoint) else train_endpoint
        train_window = get_window(
            pd.Series(index=y[y <= train_end], dtype=y.dtype),
            window_length=window_length,
        ).index
    # when given train end point is negative no training window is provided
    null = 0 if is_int(train_endpoint) else pd.Timestamp(0)
    if train_endpoint < null:
        train_window = []
    training_window = y.get_indexer(train_window)
    return training_window
