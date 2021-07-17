#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "Piyush Gade"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_SPS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.utils._testing.forecasting import _assert_correct_pred_time_index
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
def test_strategy_last_seasonal(fh, sp):
    f = NaiveForecaster(strategy="last", sp=sp)
    f.fit(y_train)
    y_pred = f.predict(fh)

    # check predicted index
    _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)

    # check values
    fh = check_fh(fh)  # get well formatted fh
    reps = int(np.ceil(max(fh) / sp))
    expected = np.tile(y_train.iloc[-sp:], reps=reps)[fh - 1]
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("sp", TEST_SPS)
@pytest.mark.parametrize("window_length", [*TEST_WINDOW_LENGTHS, None])
def test_strategy_mean_seasonal(fh, sp, window_length):
    if (window_length is not None and window_length > sp) or (window_length is None):
        f = NaiveForecaster(strategy="mean", sp=sp, window_length=window_length)
        f.fit(y_train)
        y_pred = f.predict(fh)

        # check predicted index
        _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)

        if window_length is None:
            window_length = len(y_train)

        # check values
        fh = check_fh(fh)  # get well formatted fh
        reps = int(np.ceil(max(fh) / sp))
        last_window = y_train.iloc[-window_length:].to_numpy().astype(float)
        last_window = np.pad(
            last_window,
            (sp - len(last_window) % sp, 0),
            "constant",
            constant_values=np.nan,
        )

        last_window = last_window.reshape(int(np.ceil(len(last_window) / sp)), sp)
        expected = np.tile(np.nanmean(last_window, axis=0), reps=reps)[fh - 1]
        np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("n_seasons", [1, 3])
@pytest.mark.parametrize("sp", TEST_SPS)
def test_strategy_mean_seasonal_simple(n_seasons, sp):
    # create 2d matrix, rows are different seasons, columns time points of
    # each season
    values = np.random.normal(size=(n_seasons, sp))
    y = pd.Series(values.ravel())

    expected = values.mean(axis=0)
    assert expected.shape == (sp,)

    f = NaiveForecaster(strategy="mean", sp=sp)
    f.fit(y)
    fh = np.arange(1, sp + 1)
    y_pred = f.predict(fh)

    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", [*TEST_WINDOW_LENGTHS, None])
def test_strategy_drift_unit_slope(fh, window_length):
    # drift strategy for constant slope 1
    if window_length != 1:
        f = NaiveForecaster(strategy="drift", window_length=window_length)
        f.fit(y_train)
        y_pred = f.predict(fh)

        if window_length is None:
            window_length = len(y_train)

        # get well formatted fh values
        fh = check_fh(fh)

        expected = y_train.iloc[-1] + np.arange(0, max(fh) + 1)[fh]
        np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", [*TEST_WINDOW_LENGTHS, None])
def test_strategy_drift_flat_line(fh, window_length):
    # test for flat time series data
    if window_length != 1:
        y_train = pd.Series(np.ones(20))
        f = NaiveForecaster(strategy="drift", window_length=window_length)
        f.fit(y_train)
        y_pred = f.predict(fh)

        if window_length is None:
            window_length = len(y_train)

        # get well formatted fh values
        fh = check_fh(fh)
        expected = np.ones(len(fh))

        np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", [*TEST_WINDOW_LENGTHS, None])
def test_strategy_drift_window_length(fh, window_length):
    # test for checking if window_length is properly working
    if window_length != 1:
        if window_length is None:
            window_length = len(y_train)

        values = np.random.normal(size=window_length)
        y = pd.Series(values)
        f = NaiveForecaster(strategy="drift", window_length=window_length)
        f.fit(y)
        y_pred = f.predict(fh)

        slope = (values[-1] - values[0]) / (window_length - 1)

        # get well formatted fh values
        fh = check_fh(fh)
        expected = values[-1] + slope * fh

        np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("window_length", list({4, 5, *TEST_WINDOW_LENGTHS}))
@pytest.mark.parametrize("sp", list({3, 4, 8, *TEST_SPS}))
def test_strategy_mean_seasonal_additional_combinations(n, window_length, sp):
    """Check time series of n * window_length with a 1:n-1 train/test split,
    for different combinations of the period and seasonal periodicity.
    The time series contains perfectly cyclic data.
    """

    # given <window_length> hours of data with a seasonal periodicity of <sp> hours
    freq = pd.Timedelta("1H")
    data = pd.Series(
        index=pd.date_range(
            "2021-06-01 00:00", periods=n * window_length, freq=freq, closed="left"
        ),
        data=([float(i) for i in range(1, sp + 1)] * n * window_length)[
            : n * window_length
        ],
    )

    # Split into train and test data
    train_data = data[:window_length]
    test_data = data[window_length:]

    # Forecast data does not retain the original frequency
    test_data.index.freq = None

    # For example, for n=2, periods=4 and sp=3:

    # print(train_data)
    # 2021-06-01 00:00:00    1.0
    # 2021-06-01 01:00:00    2.0
    # 2021-06-01 02:00:00    3.0
    # 2021-06-01 03:00:00    1.0
    # Freq: H, dtype: int64

    # print(test_data)
    # 2021-06-01 04:00:00    2.0  # (value of 3 hours earlier)
    # 2021-06-01 05:00:00    3.0  # (value of 3 hours earlier)
    # 2021-06-01 06:00:00    1.0  # (mean value of 3 and 6 hours earlier)
    # 2021-06-01 07:00:00    2.0  # (value of 6 hours earlier)
    # dtype: float64

    # let's forecast the next <2 x period> hours with a periodicity of <sp> hours
    fh = ForecastingHorizon(test_data.index, is_relative=False)
    model = NaiveForecaster(strategy="mean", sp=sp)
    model.fit(train_data)
    forecast_data = model.predict(fh)

    if sp < window_length:
        # We expect a perfect forecast given our perfectly cyclic data
        pd.testing.assert_series_equal(forecast_data, test_data)
    else:
        # We expect a few forecasts yield NaN values
        for i in range(1 + len(test_data) // sp):
            test_data[i * sp : i * sp + sp - window_length] = np.nan
        pd.testing.assert_series_equal(forecast_data, test_data)
