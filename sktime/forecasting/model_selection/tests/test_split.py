#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.model_selection import CutoffSplitter
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_FHS
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_STEP_LENGTHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.forecasting.tests._config import TEST_YS
from sktime.forecasting.tests._config import VALID_INDEX_FH_COMBINATIONS
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.series import _make_series
from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh

# generate random series
n_timepoints = 30
CUTOFFS = [
    np.array([21, 22]),
    np.array([3, 7, 10]),
]


def check_window_properties(windows, allow_empty=False):
    """Helper function to test common properties of windows"""
    assert isinstance(windows, list)
    for window in windows:
        assert isinstance(window, np.ndarray)
        assert all([is_int(idx) for idx in window])
        assert window.ndim == 1
        if not allow_empty:
            assert len(window) > 0


def check_cutoff_properties(cutoffs):
    """Helper function to test common properties of cutoffs"""
    assert isinstance(cutoffs, np.ndarray)
    assert all([is_int(cutoff) for cutoff in cutoffs])
    assert cutoffs.ndim == 1
    assert len(cutoffs) > 0


def check_n_splits_properties(n_splits):
    """Helper function to test common properties of n_splits"""
    assert is_int(n_splits)
    assert n_splits > 0


def generate_and_check_windows(y, cv):
    """Helper function to generate and test output from cv iterators"""
    training_windows = []
    test_windows = []
    for training_window, test_window in cv.split(y):
        # check if indexing works
        _ = y.iloc[training_window]
        _ = y.iloc[test_window]

        # keep windows for more checks
        training_windows.append(training_window)
        test_windows.append(test_window)

    # check windows, allow empty for cv starting at fh rather than window
    check_window_properties(training_windows, allow_empty=True)
    check_window_properties(test_windows, allow_empty=True)

    # check cutoffs
    cutoffs = cv.get_cutoffs(y)
    check_cutoff_properties(cutoffs)

    # check n_splits
    n_splits = cv.get_n_splits(y)
    check_n_splits_properties(n_splits)
    assert n_splits == len(training_windows) == len(test_windows) == len(cutoffs)

    return training_windows, test_windows, n_splits, cutoffs


def get_n_incomplete_windows(windows, window_length):
    n = 0
    for window in windows:
        if len(window) < window_length:
            n += 1
    return n


def check_windows_dimensions(windows, n_incomplete_windows, window_length):
    assert n_incomplete_windows == get_n_incomplete_windows(windows, window_length)

    # check incomplete windows
    if n_incomplete_windows > 1:
        incomplete_windows = windows[:n_incomplete_windows]
        check_incomplete_windows_dimensions(
            incomplete_windows, n_incomplete_windows, window_length
        )

    # check complete windows
    n_complete_windows = len(windows) - n_incomplete_windows
    if n_complete_windows > 1:
        complete_training_windows = windows[n_incomplete_windows:]
        check_complete_windows_dimensions(
            complete_training_windows, n_complete_windows, window_length
        )


def check_complete_windows_dimensions(windows, n_windows, window_length):
    windows = np.vstack(windows)
    assert windows.shape == (n_windows, window_length)


def check_incomplete_windows_dimensions(windows, n_windows, max_window_length):
    assert len(windows) == n_windows
    assert all(len(window) < max_window_length for window in windows)


def check_test_windows(windows, fh, cutoffs):
    n_incomplete_windows = get_n_incomplete_windows(windows, len(check_fh(fh)))
    check_windows_dimensions(windows, n_incomplete_windows, len(check_fh(fh)))

    np.testing.assert_array_equal(
        cutoffs[n_incomplete_windows:],
        np.vstack(windows[n_incomplete_windows:])[:, 0] - np.min(check_fh(fh)),
    )


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_single_window_split(y, fh, window_length):
    cv = SingleWindowSplitter(fh=fh, window_length=window_length)
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(
        y, cv
    )

    training_window = training_windows[0]
    test_window = test_windows[0]

    assert n_splits == 1
    assert training_window.shape[0] == window_length
    assert training_window[-1] == cutoffs[0]
    assert test_window.shape[0] == len(check_fh(fh))
    np.testing.assert_array_equal(test_window, training_window[-1] + check_fh(fh))


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("cutoffs", CUTOFFS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_manual_window_split(y, cutoffs, fh, window_length):
    # initiate rolling window cv iterator
    cv = CutoffSplitter(cutoffs, fh=fh, window_length=window_length)

    # generate and keep splits
    training_windows, test_windows, n_splits, _ = generate_and_check_windows(y, cv)

    # check cutoffs
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))

    # check training windows
    n_incomplete_windows = get_n_incomplete_windows(training_windows, window_length)
    check_windows_dimensions(training_windows, n_incomplete_windows, window_length)

    # check test windows
    check_test_windows(test_windows, fh, cutoffs)


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_split_start_with_window(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        start_with_window=True,
    )

    # generate and keep splits
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(
        y, cv
    )

    # check training windows
    n_incomplete_windows = 0  # infer expected number of incomplete windows
    check_windows_dimensions(training_windows, n_incomplete_windows, window_length)

    # check training windows values
    training_windows = np.vstack(training_windows)

    # check against cutoffs
    np.testing.assert_array_equal(cutoffs, training_windows[:, -1])

    # check values of first window
    np.testing.assert_array_equal(training_windows[0, :], np.arange(window_length))

    # check against step length
    np.testing.assert_array_equal(
        training_windows[:, 0] // step_length, np.arange(n_splits)
    )

    # check test windows
    check_test_windows(test_windows, fh, cutoffs)


@pytest.mark.parametrize("y", TEST_YS)
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
def test_sliding_window_split_start_with_fh(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=window_length,
        step_length=step_length,
        start_with_window=False,
    )

    # generate and keep splits
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(
        y, cv
    )

    # check first windows
    assert len(training_windows[0]) == 0
    assert len(training_windows[1]) == min(step_length, window_length)

    # check training windows
    n_incomplete_windows = np.int(
        np.ceil(window_length / step_length)
    )  # infer expected number of incomplete
    # windows
    check_windows_dimensions(training_windows, n_incomplete_windows, window_length)

    # check test windows
    check_test_windows(test_windows, fh, cutoffs)


@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("values", TEST_OOS_FHS)
def test_split_by_fh(index_type, fh_type, is_relative, values):
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
