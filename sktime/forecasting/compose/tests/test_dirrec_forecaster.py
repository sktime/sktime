#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "Ayushmaan Seth"]
import numpy as np
import pytest
from sktime.forecasting.compose._reduce import ReducedRegressionForecaster
from sktime.forecasting.tests import TEST_OOS_FHS
from sktime.forecasting.tests import TEST_WINDOW_LENGTHS
from sktime.utils._testing import make_forecasting_problem
from sklearn.dummy import DummyRegressor

y_train = make_forecasting_problem()


def extract_expected_mean(y_train, fh_length, window_length):
    start_fh = fh_length - 1
    expected = []
    while start_fh > 0:
        expected.append(np.mean(y_train.iloc[window_length:-(start_fh)]))
        start_fh -= 1
        window_length += 1

    expected.append(np.mean(y_train.iloc[window_length + start_fh:]))
    return np.asarray(expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_strategy_direct(fh, window_length):
    regressor = DummyRegressor(strategy="mean")
    forecaster = ReducedRegressionForecaster(regressor=regressor,
                                             window_length=window_length,
                                             strategy="direct")
    forecaster.fit(y_train, fh)
    y_pred = forecaster.predict(fh)
    actual = y_pred
    if isinstance(fh, int):
        expected = extract_expected_mean(y_train, fh, window_length)
    else:
        expected = extract_expected_mean(y_train, fh.shape[0], window_length)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_strategy_recursive(fh, window_length):
    regressor = DummyRegressor()
    forecaster = ReducedRegressionForecaster(regressor=regressor,
                                             window_length=window_length,
                                             strategy="recursive")
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    actual = np.unique(y_pred.to_numpy())
    expected = y_train.iloc[window_length:].mean()
    assert actual == expected


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_strategy_dirrec(fh, window_length):
    regressor = DummyRegressor(strategy="mean")
    forecaster = ReducedRegressionForecaster(regressor=regressor,
                                             window_length=window_length,
                                             strategy="dirrec")
    forecaster.fit(y_train, fh)
    y_pred = forecaster.predict(fh)
    actual = y_pred
    if isinstance(fh, int):
        expected = extract_expected_mean(y_train, fh, window_length)
    else:
        expected = extract_expected_mean(y_train, fh.shape[0], window_length)

    np.testing.assert_array_equal(actual, expected)
