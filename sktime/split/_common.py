# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for time series splitters."""

import pandas as pd


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
