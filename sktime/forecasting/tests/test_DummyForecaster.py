#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.dummy import DummyForecaster
from sktime.utils.validation.forecasting import validate_fh

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
def test_strategy_last(fh):
    f = DummyForecaster(strategy="last")
    f.fit(y_train)
    y_pred = f.predict(fh)
    expected = np.repeat(y_train.iloc[-1], len(f.fh))
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("window_length", [None, 3, 5])
def test_strategy_mean(fh, window_length):
    f = DummyForecaster(strategy="mean", window_length=window_length)
    f.fit(y_train)
    y_pred = f.predict(fh)

    if window_length is None:
        window_length = len(y_train)

    expected = np.repeat(y_train.iloc[-window_length:].mean(), len(f.fh))
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("sp", [3, 7, 12])
def test_strategy_seasonal_last(fh, sp):
    f = DummyForecaster(strategy="seasonal_last", sp=sp)
    f.fit(y_train)
    y_pred = f.predict(fh)

    fh = validate_fh(fh)  # get well formatted fh
    reps = np.int(np.ceil(max(fh) / sp))
    expected = np.tile(y_train.iloc[-sp:], reps=reps)[fh - np.min(fh)]
    np.testing.assert_array_equal(y_pred, expected)
