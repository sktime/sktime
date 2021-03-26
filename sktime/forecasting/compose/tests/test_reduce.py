#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal", "Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import DirectTabularRegressionForecaster
from sktime.forecasting.compose import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose import MultioutputTabularRegressionForecaster
from sktime.forecasting.compose import RecursiveTabularRegressionForecaster
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection.tests.test_split import _get_windows
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.transformations.panel.reduce import Tabularizer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.validation.forecasting import check_fh

N_TIMEPOINTS = [13, 17]
N_VARIABLES = [1, 3]
FH = ForecastingHorizon(1)


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("scitype", ["tabular-regressor", "time-series-regressor"])
def test_against_sliding_window_cv(n_timepoints, window_length, fh, scitype):
    fh = check_fh(fh)
    y = pd.Series(_make_y(0, n_timepoints))
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length)
    xa, ya = _get_windows(cv, y)
    yb, xb = _sliding_window_transform(y, window_length, fh, scitype=scitype)
    np.testing.assert_array_equal(ya, yb)
    if scitype == "time-series-regressor":
        xb = xb.squeeze(axis=1)

    np.testing.assert_array_equal(xa, xb)


def _make_y_X(n_timepoints, n_variables):
    # We generate y and X values so that the y should always be greater
    # than its lagged values and the lagged and contemporaneous values of the
    # exogenous variables X
    assert n_variables < 10

    base = np.arange(n_timepoints)
    y = pd.Series(base + n_variables / 10)

    if n_variables > 1:
        X = np.column_stack([base + i / 10 for i in range(1, n_variables)])
        X = pd.DataFrame(X)
    else:
        X = None

    return y, X


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_tranform_tabular(n_timepoints, window_length, n_variables, fh):
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, fh=fh, X=X, scitype="tabular-regressor"
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values for first step in fh.
    actual = yt[:, 0]
    start = window_length + fh[0] - 1
    end = start + n_timepoints - window_length - fh_max + 1
    expected = y[np.arange(start, end)]
    np.testing.assert_array_equal(actual, expected)

    # The transformed Xt array contains lagged values for each variable, excluding the
    # n_variables + 1 contemporaneous values for the exogenous variables.
    # assert Xt.shape == (yt.shape[0], (window_length * n_variables) + n_variables - 1)
    assert Xt.shape == (yt.shape[0], window_length * n_variables)
    assert np.all(Xt < yt[:, [0]])


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_variables", N_VARIABLES)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_sliding_window_tranform_panel(n_timepoints, window_length, n_variables, fh):
    y, X = _make_y_X(n_timepoints=n_timepoints, n_variables=n_variables)
    fh = check_fh(fh, enforce_relative=True)
    fh_max = fh[-1]
    effective_window_length = window_length + fh_max - 1

    yt, Xt = _sliding_window_transform(
        y, window_length=window_length, X=X, fh=fh, scitype="time-series-regressor"
    )
    assert yt.shape == (n_timepoints - effective_window_length, len(fh))

    # Check y values.
    actual = yt[:, 0]
    start = window_length + fh[0] - 1
    end = start + n_timepoints - window_length - fh_max + 1
    expected = y[np.arange(start, end)]
    np.testing.assert_array_equal(actual, expected)

    # Given the input data, all of the value in the transformed Xt array should be
    # smaller than the transformed yt target array.
    assert Xt.shape == (yt.shape[0], n_variables, window_length)
    assert np.all(Xt < yt[:, np.newaxis, [0]])


def _make_y(start, end, method="linear-trend", slope=1):
    # generate test data
    if method == "linear-trend":
        y = np.arange(start, end) * slope
    else:
        raise ValueError("`method` not understood")
    return y


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("strategy", ["recursive", "direct", "multioutput"])
@pytest.mark.parametrize(
    "regressor, scitype",
    [
        (LinearRegression(), "tabular-regressor"),
        (make_pipeline(Tabularizer(), LinearRegression()), "time-series-regressor"),
    ],
)
@pytest.mark.parametrize(
    "method, slope",
    [
        ("linear-trend", 1),
        ("linear-trend", -3),
        ("linear-trend", 0),  # constant
    ],
)
def test_linear_extrapolation(
    fh, window_length, strategy, method, slope, regressor, scitype
):
    n_timepoints = 13
    y = _make_y(0, n_timepoints, method=method, slope=slope)
    y = pd.Series(y)
    fh = check_fh(fh)

    forecaster = make_reduction(
        regressor, scitype=scitype, window_length=window_length, strategy=strategy
    )
    forecaster.fit(y, fh=fh)
    actual = forecaster.predict()

    end = n_timepoints + max(fh) + 1
    expected = _make_y(n_timepoints, end, method=method, slope=slope)[fh.to_indexer()]
    np.testing.assert_almost_equal(actual, expected)


@pytest.mark.parametrize("fh", [1, 3, 5])
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("strategy", ["recursive", "direct", "multioutput"])
@pytest.mark.parametrize("scitype", ["time-series-regressor", "tabular-regressor"])
def test_dummy_regressor_mean_prediction(fh, window_length, strategy, scitype):
    # The DummyRegressor ignores the input feature data X, hence we can use it for
    # testing reduction from forecasting to both tabular and time series regression.
    # The DummyRegressor also supports the 'multioutput' strategy.
    y, X = make_forecasting_problem(make_X=True)
    fh = check_fh(fh)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)

    regressor = DummyRegressor(strategy="mean")
    forecaster = make_reduction(
        regressor, scitype=scitype, window_length=window_length, strategy=strategy
    )
    forecaster.fit(y_train, X_train, fh=fh)
    actual = forecaster.predict(X=X_test)

    if strategy == "recursive":
        # For the recursive strategy, we always use the first-step ahead as the
        # target vector in the regression problem during training, regardless of the
        # actual forecasting horizon.
        effective_window_length = window_length
    else:
        # For the other strategies, we split the data taking into account the steps
        # ahead we want to predict.
        effective_window_length = window_length + max(fh) - 1

    expected = np.mean(y_train[effective_window_length:])
    np.testing.assert_array_almost_equal(actual, expected)


def test_factory_method_recursive():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = make_reduction(regressor, scitype="tabular-regressor", strategy="recursive")
    f2 = RecursiveTabularRegressionForecaster(regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = f2.fit(y_train).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_direct():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = make_reduction(regressor, scitype="tabular-regressor", strategy="direct")
    f2 = DirectTabularRegressionForecaster(regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = f2.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_ts_recursive():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = make_reduction(
        ts_regressor, scitype="time-series-regressor", strategy="recursive"
    )
    f2 = RecursiveTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = f2.fit(y_train).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_ts_direct():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = make_reduction(
        ts_regressor, scitype="time-series-regressor", strategy="direct"
    )
    f2 = DirectTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = f2.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_multioutput_direct_tabular():
    # multioutput and direct strategies with linear regression
    # regressor should produce same predictions
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = MultioutputTabularRegressionForecaster(regressor)
    f2 = DirectTabularRegressionForecaster(regressor)

    preds1 = f1.fit(y_train, fh=fh).predict(fh)
    preds2 = f2.fit(y_train, fh=fh).predict(fh)

    # assert_almost_equal does not seem to work with pd.Series objects
    np.testing.assert_almost_equal(preds1.to_numpy(), preds2.to_numpy(), decimal=5)
