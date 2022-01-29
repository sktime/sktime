#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Test splitters."""

__author__ = ["Markus Löning", "Kutay Koralturk", "khrapovs"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.tests._config import (
    TEST_FHS,
    TEST_INITIAL_WINDOW,
    TEST_OOS_FHS,
    TEST_STEP_LENGTHS,
    TEST_WINDOW_LENGTHS,
    TEST_YS,
    VALID_INDEX_FH_COMBINATIONS,
)
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.series import _make_series
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.validation import is_int, is_timedelta_or_date_offset
from sktime.utils.validation.forecasting import check_fh

N_TIMEPOINTS = 30
CUTOFFS = [np.array([21, 22]), np.array([3, 7, 10])]


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
    assert np.issubdtype(cutoffs.dtype, np.integer)
    assert cutoffs.ndim == 1
    assert len(cutoffs) > 0


def _check_n_splits(n_splits):
    assert is_int(n_splits)
    assert n_splits > 0


def _check_cutoffs_against_test_windows(cutoffs, windows, fh):
    # We check for the last value. Some windows may be incomplete, with no first
    # value, whereas the last value will always be there.
    fh = check_fh(fh)
    expected = np.array([window[-1] - fh[-1] for window in windows])
    np.testing.assert_array_equal(cutoffs, expected)


def _check_cutoffs_against_train_windows(cutoffs, windows):
    # Cutoffs should always be the last values of the train windows.
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
    _check_cutoffs_against_test_windows(cutoffs, test_windows, cv.fh)
    _check_cutoffs_against_train_windows(cutoffs, train_windows)

    n_splits = cv.get_n_splits(y)
    _check_n_splits(n_splits)
    assert n_splits == len(train_windows) == len(test_windows) == len(cutoffs)

    return train_windows, test_windows, cutoffs, n_splits


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_single_window_splitter(y, fh, window_length):
    """Test SingleWindowSplitter."""
    cv = SingleWindowSplitter(fh=fh, window_length=window_length)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)

    train_window = train_windows[0]
    test_window = test_windows[0]
    assert n_splits == 1
    assert train_window.shape[0] == _coerce_duration_to_int(
        duration=window_length, freq="D"
    )
    assert test_window.shape[0] == len(check_fh(fh))

    np.testing.assert_array_equal(test_window, train_window[-1] + check_fh(fh))


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
def test_single_window_splitter_default_window_length(y, fh):
    """Test SingleWindowSplitter."""
    cv = SingleWindowSplitter(fh=fh)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)

    train_window = train_windows[0]
    test_window = test_windows[0]

    assert n_splits == 1
    assert test_window.shape[0] == len(check_fh(fh))

    fh = cv.get_fh()
    if fh.is_all_in_sample():
        assert train_window.shape[0] == len(y)
    else:
        assert train_window.shape[0] == len(y) - fh.max()

    np.testing.assert_array_equal(test_window, train_window[-1] + check_fh(fh))


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("cutoffs", CUTOFFS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_cutoff_window_splitter(y, cutoffs, fh, window_length):
    """Test CutoffSplitter."""
    cv = CutoffSplitter(cutoffs, fh=fh, window_length=window_length)
    train_windows, test_windows, cutoffs, n_splits = _check_cv(cv, y)
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_splitter(y, fh, window_length, step_length):
    """Test SlidingWindowSplitter."""
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        start_with_window=True,
    )
    train_windows, test_windows, _, n_splits = _check_cv(cv, y)

    assert np.vstack(train_windows).shape == (
        n_splits,
        _coerce_duration_to_int(duration=window_length, freq="D"),
    )
    assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))


def _args_are_all_of_the_same_type(*args) -> bool:
    return all([type(args[0]) == type(arg) for arg in args])


def _windows_are_incompatible(initial_window, window_length) -> bool:
    return (
        is_timedelta_or_date_offset(x=initial_window)
        and not is_timedelta_or_date_offset(x=window_length)
    ) or (
        is_timedelta_or_date_offset(x=window_length)
        and not is_timedelta_or_date_offset(x=initial_window)
    )


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("initial_window", TEST_INITIAL_WINDOW)
def test_sliding_window_splitter_with_initial_window(
    y, fh, window_length, step_length, initial_window
):
    """Test SlidingWindowSplitter."""
    if _windows_are_incompatible(initial_window, window_length):
        pytest.skip(
            "Incompatible initial_window and window_length are tested elsewhere."
        )
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        initial_window=initial_window,
        start_with_window=True,
    )
    train_windows, test_windows, _, n_splits = _check_cv(cv, y)

    assert train_windows[0].shape[0] == _coerce_duration_to_int(
        duration=initial_window, freq="D"
    )
    assert np.vstack(train_windows[1:]).shape == (
        n_splits - 1,
        _coerce_duration_to_int(duration=window_length, freq="D"),
    )
    assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("initial_window", TEST_INITIAL_WINDOW)
def test_sliding_window_splitter_with_incompatible_initial_window_and_window_length(
    y, fh, window_length, step_length, initial_window
):
    """Test SlidingWindowSplitter with incompatible initial_window and window_length."""
    if not _windows_are_incompatible(initial_window, window_length):
        pytest.skip("Compatible initial_window and window_length are tested elsewhere.")
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        initial_window=initial_window,
        start_with_window=True,
    )
    match = "The `initial_window` and `window_length` types are incompatible"
    with pytest.raises(ValueError, match=match):
        _check_cv(cv, y)


def _get_n_incomplete_windows(window_length, step_length) -> int:
    return int(
        np.ceil(
            _coerce_duration_to_int(duration=window_length, freq="D")
            / _coerce_duration_to_int(duration=step_length, freq="D")
        )
    )


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_splitter_start_with_empty_window(
    y, fh, window_length, step_length
):
    """Test SlidingWindowSplitter."""
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        start_with_window=False,
    )
    train_windows, test_windows, _, n_splits = _check_cv(cv, y, allow_empty_window=True)

    assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))

    n_incomplete = _get_n_incomplete_windows(window_length, step_length)
    train_windows = train_windows[n_incomplete:]

    assert np.vstack(train_windows).shape == (
        n_splits - n_incomplete,
        _coerce_duration_to_int(duration=window_length, freq="D"),
    )


def test_sliding_window_splitter_initial_window_start_with_empty_window_raises_error():
    """Test SlidingWindowSplitter."""
    y = _make_series()
    cv = SlidingWindowSplitter(
        fh=1,
        initial_window=15,
        start_with_window=False,
    )
    message = "`start_with_window` must be True if `initial_window` is given"
    with pytest.raises(ValueError, match=message):
        next(cv.split(y))


def test_sliding_window_splitter_initial_window_smaller_than_window_raise_error():
    """Test SlidingWindowSplitter."""
    y = _make_series()
    cv = SlidingWindowSplitter(
        fh=1,
        window_length=10,
        initial_window=5,
    )
    message = "`initial_window` must greater than `window_length`"
    with pytest.raises(ValueError, match=message):
        next(cv.split(y))


def _check_expanding_windows(windows):
    n_splits = len(windows)
    for i in range(1, n_splits):
        current = windows[i]
        previous = windows[i - 1]

        assert current.shape[0] > previous.shape[0]
        assert current[0] == previous[0]
        assert current[-1] > previous[-1]


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("initial_window", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_expanding_window_splitter_start_with_empty_window(
    y, fh, initial_window, step_length
):
    """Test ExpandingWindowSplitter."""
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=initial_window,
        step_length=step_length,
        start_with_window=True,
    )
    train_windows, test_windows, _, n_splits = _check_cv(cv, y)
    assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))

    n_incomplete = _get_n_incomplete_windows(initial_window, step_length)
    train_windows = train_windows[n_incomplete:]
    _check_expanding_windows(train_windows)


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("initial_window", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_expanding_window_splitter(y, fh, initial_window, step_length):
    """Test ExpandingWindowSplitter."""
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=initial_window,
        step_length=step_length,
        start_with_window=True,
    )
    train_windows, test_windows, _, n_splits = _check_cv(cv, y)
    assert np.vstack(test_windows).shape == (n_splits, len(check_fh(fh)))
    assert train_windows[0].shape[0] == _coerce_duration_to_int(
        duration=initial_window, freq="D"
    )
    _check_expanding_windows(train_windows)


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


@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("values", TEST_OOS_FHS)
def test_split_by_fh(index_type, fh_type, is_relative, values):
    """Test temporal_train_test_split."""
    y = _make_series(20, index_type=index_type)
    cutoff = y.index[10]
    fh = _make_fh(cutoff, values, fh_type, is_relative)
    split = temporal_train_test_split(y, fh=fh)
    _check_train_test_split_y(fh, split)


def _check_train_test_split_y(fh, split):
    assert len(split) == 2

    train, test = split
    assert isinstance(train, pd.Series)
    assert isinstance(test, pd.Series)
    assert set(train.index).isdisjoint(test.index)
    for test_timepoint in test.index:
        assert np.all(train.index < test_timepoint)
    assert len(test) == len(fh)
    assert len(train) > 0

    cutoff = train.index[-1]
    np.testing.assert_array_equal(test.index, fh.to_absolute(cutoff).to_numpy())
