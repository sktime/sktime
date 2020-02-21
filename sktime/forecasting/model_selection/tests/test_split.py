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
CUTOFF_POINTS = [
    np.array([21, 22]),
    np.array([40, 45]),
    np.array([-45, -40])
]

ALL_FHS = DEFAULT_FHS + DEFAULT_INSAMPLE_FHS


@pytest.mark.parametrize("y, cutoff_points", [(y, cutoff_points) for y, cutoff_points in zip(YS, CUTOFF_POINTS)])
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
def test_manual_window(y, cutoff_points, fh, window_length):
    # initiate rolling window cv iterator
    cv = ManualWindowSplitter(cutoff_points, fh=fh, window_length=window_length)

    # generate and keep splits
    inputs = []
    outputs = []
    for i, o in cv.split(y):
        inputs.append(i)
        outputs.append(o)
    inputs = np.vstack(inputs)
    outputs = np.vstack(outputs)

    ns = cv.get_n_splits(y)
    assert ns == len(cutoff_points)
    assert inputs.shape == (ns, window_length)  # check window length
    assert outputs.shape == (ns, len(check_fh(fh)))  # check fh

    # check if last values of input window are split points
    # comparing relative indices returned by cv iterator with absolute split points
    np.testing.assert_array_equal(y.iloc[inputs[:, -1]].values, y.loc[cutoff_points].values)


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("fh", ALL_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_sliding_window(y, fh, window_length, step_length):
    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)

    # generate and keep splits
    inputs = []
    outputs = []
    for i, o in cv.split(y):
        inputs.append(i)
        outputs.append(o)
    inputs = np.vstack(inputs)
    outputs = np.vstack(outputs)

    ns = cv.get_n_splits(y)

    np.testing.assert_array_equal(inputs[0, :], np.arange(window_length))  # check first window
    np.testing.assert_array_equal(inputs[:, 0] // step_length, np.arange(ns))  # check step length
    assert inputs.shape == (ns, window_length)  # check window length
    assert outputs.shape == (ns, len(check_fh(fh)))  # check fh
