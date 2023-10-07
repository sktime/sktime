# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for splitters."""

__author__ = ["mloning", "kkoralturk", "khrapovs", "fkiraly"]

import numpy as np
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation import (
    array_is_datetime64,
    array_is_int,
    array_is_timedelta_or_date_offset,
    is_int,
)
from sktime.utils.validation.forecasting import check_fh


def _get_windows(cv, y):
    train_windows = []
    test_windows = []

    n_splits = 0
    for train, test in cv.split(y):
        n_splits += 1
        train_windows.append(train)
        test_windows.append(test)
    assert n_splits == cv.get_n_splits(y)

    return train_windows, test_windows


def _check_windows(windows, allow_empty_window=False):
    assert isinstance(windows, list)
    for window in windows:
        assert isinstance(window, np.ndarray)
        assert np.issubdtype(window.dtype, np.integer)
        assert window.ndim == 1
        if not allow_empty_window:
            assert len(window) > 0


def _check_cutoffs(cutoffs):
    assert isinstance(cutoffs, np.ndarray)
    assert array_is_int(cutoffs) or array_is_datetime64(cutoffs)
    assert cutoffs.ndim == 1
    assert len(cutoffs) > 0


def _check_n_splits(n_splits):
    assert is_int(n_splits)
    assert n_splits > 0


def _check_cutoffs_against_test_windows(cutoffs, windows, fh, y):
    # We check for the last value. Some windows may be incomplete, with no first
    # value, whereas the last value will always be there.
    fh = check_fh(fh)
    if is_int(fh[-1]):
        expected = np.array([window[-1] - fh[-1] for window in windows])
    elif array_is_timedelta_or_date_offset(fh):
        expected = np.array([])
        for window in windows:
            arg = y.index[window[-1]] - fh[-1]
            val = y.index.get_loc(arg) if arg >= y.index[0] else -1
            expected = np.append(expected, val)
    else:
        raise ValueError(f"Provided `fh` type is not supported: {type(fh[-1])}")
    np.testing.assert_array_equal(cutoffs, expected)


def _check_cutoffs_against_train_windows(cutoffs, windows, y):
    # Cutoffs should always be the last values of the train windows.
    assert array_is_int(cutoffs)
    actual = np.array([window[-1] for window in windows[1:]])
    np.testing.assert_array_equal(actual, cutoffs[1:])

    # We treat the first window separately, since it may be empty when setting
    # `start_with_window=False`.
    if len(windows[0]) > 0:
        np.testing.assert_array_equal(windows[0][-1], cutoffs[0])


def _check_cv(cv, y, allow_empty_window=False):
    train_windows, test_windows = _get_windows(cv, y)
    _check_windows(train_windows, allow_empty_window=allow_empty_window)
    _check_windows(test_windows, allow_empty_window=allow_empty_window)

    cutoffs = cv.get_cutoffs(y)
    _check_cutoffs(cutoffs)
    _check_cutoffs_against_test_windows(cutoffs, test_windows, cv.fh, y)
    _check_cutoffs_against_train_windows(cutoffs, train_windows, y)

    n_splits = cv.get_n_splits(y)
    _check_n_splits(n_splits)
    assert n_splits == len(train_windows) == len(test_windows) == len(cutoffs)

    return train_windows, test_windows, cutoffs, n_splits


def _get_n_incomplete_windows(window_length, step_length) -> int:
    return int(
        np.ceil(
            _coerce_duration_to_int(duration=window_length, freq="D")
            / _coerce_duration_to_int(duration=step_length, freq="D")
        )
    )


@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
def test_window_splitter_in_sample_fh_smaller_than_window_length(CV):
    """Test WindowSplitter."""
    y = np.arange(10)
    fh = ForecastingHorizon([-2, 0])
    window_length = 3
    cv = CV(fh, window_length)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(test_windows[0], np.array([0, 2]))
    np.testing.assert_array_equal(train_windows[0], np.array([0, 1, 2]))


@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
def test_window_splitter_in_sample_fh_greater_than_window_length(CV):
    """Test WindowSplitter."""
    y = np.arange(10)
    fh = ForecastingHorizon([-5, -3])
    window_length = 3
    cv = CV(fh, window_length)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(test_windows[0], np.array([0, 2]))
    np.testing.assert_array_equal(train_windows[0], np.array([3, 4, 5]))
