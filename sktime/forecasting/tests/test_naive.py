#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

__author__ = "Markus Löning"

import numpy as np
import pandas as pd
import pytest
from sktime.forecasting.naive import NaiveForecaster
<<<<<<< HEAD
from sktime.forecasting.tests import TEST_OOS_FHS, TEST_SPS, TEST_WINDOW_LENGTHS
=======
from sktime.forecasting.tests import TEST_OOS_FHS
from sktime.forecasting.tests import TEST_SPS
from sktime.forecasting.tests import TEST_WINDOW_LENGTHS
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
from sktime.utils.validation.forecasting import check_fh

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_strategy_last(fh):
    f = NaiveForecaster(strategy="last")
    f.fit(y_train)
    y_pred = f.predict(fh)
    expected = np.repeat(y_train.iloc[-1], len(f.fh))
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_strategy_mean(fh, window_length):
    f = NaiveForecaster(strategy="mean", window_length=window_length)
    f.fit(y_train)
    y_pred = f.predict(fh)

    if window_length is None:
        window_length = len(y_train)

    expected = np.repeat(y_train.iloc[-window_length:].mean(), len(f.fh))
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("sp", TEST_SPS)
def test_strategy_seasonal_last(fh, sp):
<<<<<<< HEAD
    f = NaiveForecaster(strategy="last", sp=sp)
=======
    f = NaiveForecaster(strategy="seasonal_last", sp=sp)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
    f.fit(y_train)
    y_pred = f.predict(fh)

    # check predicted index
<<<<<<< HEAD
    np.testing.assert_array_equal(y_train.index[-1] + check_fh(fh), y_pred.index)
=======
    np.testing.assert_array_equal(y_train.index[-1] + check_fh(fh),
                                  y_pred.index)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

    # check values
    fh = check_fh(fh)  # get well formatted fh
    reps = np.int(np.ceil(max(fh) / sp))
    expected = np.tile(y_train.iloc[-sp:], reps=reps)[fh - 1]
    np.testing.assert_array_equal(y_pred, expected)
<<<<<<< HEAD

@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_strategy_seasonal_mean(fh, sp, window_length):
    f = NaiveForecaster(strategy="last", sp=sp, window_length=window_length)
    f.fit(y_train)
    y_pred = f.predict(fh)

    # check predicted index
    np.testing.assert_array_equal(y_train.index[-1] + check_fh(fh), y_pred.index)

    if window_length is None:
        window_length = len(y_train)

    if window_length > sp:
        # check values
        fh = check_fh(fh)  # get well formatted fh
        reps = np.int(np.ceil(max(fh) / sp))
        window = y_train.iloc[-window_length:]

        for i in range(sp):
            window.at[window[window.index % sp == i].index[-1]] = \
                window[window.index % sp == i].mean()

        expected = np.tile(window.iloc[-sp:].to_numpy(), reps = reps)[fh - 1]
        np.testing.assert_array_equal(y_pred, expected)
=======
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
