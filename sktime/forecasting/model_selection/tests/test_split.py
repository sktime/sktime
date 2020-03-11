#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from sktime.forecasting.model_selection import SingleWindowSplit
from sktime.forecasting.model_selection import SlidingWindowSplitter, ManualWindowSplitter
from sktime.forecasting.tests import DEFAULT_FHS, DEFAULT_INSAMPLE_FHS, DEFAULT_STEP_LENGTHS, DEFAULT_WINDOW_LENGTHS
from sktime.utils.testing.forecasting import make_forecasting_problem
from sktime.utils.validation import is_int
from sktime.utils.validation.forecasting import check_fh

# generate random series
YS = [
    pd.Series(np.arange(30)),  # zero-based index
    pd.Series(np.random.random(size=30), index=np.arange(30, 60)),  # non-zero-based index
    pd.Series(np.random.random(size=30), index=np.arange(-60, -30))  # negative index
]
CUTOFFS = [
    np.array([21, 22]),
    np.array([3, 7, 10]),
    np.array([-5, -1, 0, 1, 2])
]

ALL_FHS = DEFAULT_FHS + DEFAULT_INSAMPLE_FHS


def check_windows(windows, allow_empty=False):
    """Helper function to test common properties of windows"""
    assert isinstance(windows, list)
    for window in windows:
        assert isinstance(window, np.ndarray)
        assert all([is_int(idx) for idx in window])
        assert window.ndim == 1
        if not allow_empty:
            assert len(window) > 0


def check_cutoffs(cutoffs):
    """Helper function to test common properties of cutoffs"""
    assert isinstance(cutoffs, np.ndarray)
    assert cutoffs.ndim == 1
    assert all([is_int(cutoff) for cutoff in cutoffs])


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
    check_windows(training_windows, allow_empty=True)
    check_windows(test_windows)

    # check cutoffs
    cutoffs = cv.get_cutoffs(y)
    check_cutoffs(cutoffs)

    # check n_splits
    n_splits = cv.get_n_splits(y)
    assert is_int(n_splits)
    assert n_splits == len(training_windows) == len(test_windows) == len(cutoffs)

    return training_windows, test_windows, n_splits, cutoffs


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
def test_single_window_split(y, fh, window_length):
    cv = SingleWindowSplit(fh=fh, window_length=window_length)
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(y, cv)

    training_window = training_windows[0]
    test_window = test_windows[0]

    assert n_splits == 1
    assert training_window.shape[0] == window_length
    assert training_window[-1] == cutoffs[0]
    assert test_window.shape[0] == len(check_fh(fh))
    np.testing.assert_array_equal(test_window, training_window[-1] + check_fh(fh))


@pytest.mark.parametrize("y, cutoffs", [(y, cutoffs) for y, cutoffs in zip(YS, CUTOFFS)])
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
def test_manual_window_split(y, cutoffs, fh, window_length):
    # initiate rolling window cv iterator
    cv = ManualWindowSplitter(cutoffs, fh=fh, window_length=window_length)

    # generate and keep splits
    training_windows, test_windows, n_splits, _ = generate_and_check_windows(y, cv)

    test_windows = np.vstack(test_windows)
    assert test_windows.shape == (n_splits, len(check_fh(fh)))

    # check cutoffs
    np.testing.assert_array_equal(cutoffs, cv.get_cutoffs(y))
    np.testing.assert_array_equal(cutoffs, test_windows[:, 0] - np.min(check_fh(fh)))

    # check incomplete and full windows
    n_incomplete_windows = sum(cutoffs - window_length < 0)
    if 1 < n_incomplete_windows < n_splits:
        assert len(training_windows[0]) < window_length
        training_windows = np.vstack(training_windows[n_incomplete_windows:])
        assert training_windows.shape == (n_splits - n_incomplete_windows, window_length)


@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
def test_manual_window_split_full_in_sample(window_length):
    y, y_test = make_forecasting_problem()
    cutoffs = -np.arange(len(y)) + len(y) - 2
    cv = ManualWindowSplitter(cutoffs, fh=1, window_length=window_length)

    # generate and keep splits
    training_windows, test_windows, _, _ = generate_and_check_windows(y, cv)
    assert len(training_windows[0]) == 0
    np.testing.assert_array_equal(np.hstack(test_windows), np.arange(len(y)))


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_sliding_window_split_start_with_window(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length, start_with_window=True)

    # generate and keep splits
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(y, cv)
    training_windows = np.vstack(training_windows)
    test_windows = np.vstack(test_windows)

    # check window shapes
    assert training_windows.shape == (n_splits, window_length)  # check window length
    assert test_windows.shape == (n_splits, len(check_fh(fh)))  # check fh

    # check cutoffs
    np.testing.assert_array_equal(cutoffs, training_windows[:, -1])

    # check first window
    np.testing.assert_array_equal(training_windows[0, :], np.arange(window_length))

    # check step length
    np.testing.assert_array_equal(training_windows[:, 0] // step_length, np.arange(n_splits))


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_sliding_window_split_start_with_fh(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length, start_with_window=False)

    # generate and keep splits
    training_windows, test_windows, n_splits, cutoffs = generate_and_check_windows(y, cv)

    # check first windows
    assert len(training_windows[0]) == 0
    assert len(training_windows[1]) == min(step_length, window_length)

    # check window shapes
    test_windows = np.vstack(test_windows)
    assert test_windows.shape == (n_splits, len(check_fh(fh)))

    # check cutoffs
    np.testing.assert_array_equal(cutoffs, test_windows[:, 0] - np.min(check_fh(fh)))

    # check full windows
    n_incomplete_windows = np.int(np.ceil(window_length / step_length))
    training_windows = np.vstack(training_windows[n_incomplete_windows:])
    assert training_windows.shape == (n_splits - n_incomplete_windows, window_length)
