#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"


from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.validation.forecasting import check_fh
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("window_length", [1, 3, 5])
@pytest.mark.parametrize("step_length", [1, 3, 5])
def test_splitting(fh, window_length, step_length):

    # generate random series
    idx = np.arange(100)
    s = pd.Series(idx)

    # initiate rolling window cv iterator
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)

    inputs = []
    outputs = []

    # generate and keep splits
    for i, o in cv.split(s):
        inputs.append(i)
        outputs.append(o)
    inputs = np.vstack(inputs)
    outputs = np.vstack(outputs)

    # compare actual values against expected values
    ns = cv.get_n_splits()

    # check first window
    np.testing.assert_array_equal(inputs[0, :], np.arange(window_length))

    # check window length
    assert inputs.shape == (ns, window_length)

    # check fh
    assert outputs.shape == (ns, len(check_fh(fh)))

    # check step length
    np.testing.assert_array_equal(inputs[:, 0] // step_length, np.arange(ns))


