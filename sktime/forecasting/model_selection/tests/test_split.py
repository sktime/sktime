#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from sktime.forecasting.model_selection import SlidingWindowSplitter, ManualWindowSplitter
from sktime.forecasting.tests import DEFAULT_FHS, DEFAULT_INSAMPLE_FHS, DEFAULT_STEP_LENGTHS, DEFAULT_WINDOW_LENGTHS
from sktime.utils.validation.forecasting import check_fh

# generate random series
YS = [
    pd.Series(np.arange(30)),  # zero-based index
    pd.Series(np.random.random(size=30), index=np.arange(30, 60)),  # non-zero-based index
    pd.Series(np.random.random(size=30), index=np.arange(-60, -30))  # negative index
]
CUTOFFS = [
    np.array([21, 22]),
    np.array([40, 45]),
    np.array([-45, -40])
]

ALL_FHS = DEFAULT_FHS + DEFAULT_INSAMPLE_FHS


def generate_windows(y, cv):
    training_windows = []
    test_windows = []
    for training_window, test_window in cv.split(y):
        training_windows.append(training_window)
        test_windows.append(test_window)
    return training_windows, test_windows


@pytest.mark.parametrize("y, cutoffs", [(y, cutoffs) for y, cutoffs in zip(YS, CUTOFFS)])
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
def test_manual_window_split(y, cutoffs, fh, window_length):
    # initiate rolling window cv iterator
    cv = ManualWindowSplitter(cutoffs, fh=fh, window_length=window_length)

    # generate and keep splits
    training_windows, test_windows = generate_windows(y, cv)
    inputs = np.vstack(training_windows)
    outputs = np.vstack(test_windows)

    # check number of splits
    n_splits = cv.get_n_splits(y)
    assert n_splits == len(cutoffs)

    # check window length
    assert inputs.shape == (n_splits, window_length)

    # check fh
    assert outputs.shape == (n_splits, len(check_fh(fh)))

    # check cutoffs
    np.testing.assert_array_equal(cv.get_cutoffs(y), cutoffs)
    # comparing relative indices returned by cv iterator with absolute cutoffs
    np.testing.assert_array_equal(y.iloc[inputs[:, -1]].values, y.loc[cutoffs].values)


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_sliding_window_split_start_with_window(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length, start_with_window=True)

    # generate and keep splits
    training_windows, test_windows = generate_windows(y, cv)

    training_windows = np.vstack(training_windows)
    test_windows = np.vstack(test_windows)

    n_splits = cv.get_n_splits(y)
    cutoffs = cv.get_cutoffs(y)

    # check cutoffs
    cutoff_indicies = training_windows[:, -1]
    assert n_splits == len(cutoffs)
    np.testing.assert_array_equal(y.iloc[cutoff_indicies].index.values, cutoffs)

    # check first window
    np.testing.assert_array_equal(training_windows[0, :], np.arange(window_length))

    # check step length
    np.testing.assert_array_equal(training_windows[:, 0] // step_length, np.arange(n_splits))

    # check window length
    assert training_windows.shape == (n_splits, window_length)

    # check fh
    assert test_windows.shape == (n_splits, len(check_fh(fh)))


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_sliding_window_split_start_with_fh(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length, start_with_window=False)

    # generate and keep splits
    training_windows, test_windows = generate_windows(y, cv)

    # check first windows
    assert len(training_windows[0]) == 0
    assert len(training_windows[1]) == min(step_length, window_length)

    n_splits = cv.get_n_splits(y)
    assert len(training_windows) == n_splits
    assert len(test_windows) == n_splits

    # check full windows
    n_incomplete_windows = np.int(np.ceil(window_length / step_length))
    training_windows = np.vstack(training_windows[n_incomplete_windows:])

    # check window length
    assert training_windows.shape == (n_splits - n_incomplete_windows, window_length)

    # check fh
    test_windows = np.vstack(test_windows)
    assert test_windows.shape == (n_splits, len(check_fh(fh)))
