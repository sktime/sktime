#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal", "Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import DirectTabularRegressionForecaster
from sktime.forecasting.compose import DirectTimeSeriesRegressionForecaster
from sktime.forecasting.compose import MultioutputTabularRegressionForecaster
from sktime.forecasting.compose import MultioutputTimeSeriesRegressionForecaster
from sktime.forecasting.compose import RecursiveTabularRegressionForecaster
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection.tests.test_split import _get_windows
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.regression.base import BaseRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
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
def test_sliding_window_transform_against_cv(n_timepoints, window_length, fh, scitype):
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


def test_sliding_window_transform_explicit():
    """
    testing with explicitly written down expected outputs
    intended to help future contributors understand the transformation
    """
    y = pd.Series(np.arange(9))
    X = pd.concat([y + 100, y + 200], axis=1)
    fh = ForecastingHorizon([1, 3], is_relative=True)
    window_length = 2

    yt_actual, Xt_tabular_actual = _sliding_window_transform(y, window_length, fh, X)
    _, Xt_time_series_actual = _sliding_window_transform(
        y, window_length, fh, X, scitype="time-series-regressor"
    )

    yt_expected = np.array([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0], [5.0, 7.0], [6.0, 8.0]])

    Xt_tabular_expected = np.array(
        [
            [0.0, 1.0, 100.0, 101.0, 200.0, 201.0],
            [1.0, 2.0, 101.0, 102.0, 201.0, 202.0],
            [2.0, 3.0, 102.0, 103.0, 202.0, 203.0],
            [3.0, 4.0, 103.0, 104.0, 203.0, 204.0],
            [4.0, 5.0, 104.0, 105.0, 204.0, 205.0],
        ]
    )

    Xt_time_series_expected = np.array(
        [
            [[0.0, 1.0], [100.0, 101.0], [200.0, 201.0]],
            [[1.0, 2.0], [101.0, 102.0], [201.0, 202.0]],
            [[2.0, 3.0], [102.0, 103.0], [202.0, 203.0]],
            [[3.0, 4.0], [103.0, 104.0], [203.0, 204.0]],
            [[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]],
        ]
    )

    np.testing.assert_array_equal(yt_actual, yt_expected)
    np.testing.assert_array_equal(Xt_tabular_actual, Xt_tabular_expected)
    np.testing.assert_array_equal(Xt_time_series_actual, Xt_time_series_expected)


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


_REGISTRY = [
    ("tabular-regressor", "direct", DirectTabularRegressionForecaster),
    ("tabular-regressor", "recursive", RecursiveTabularRegressionForecaster),
    ("tabular-regressor", "multioutput", MultioutputTabularRegressionForecaster),
    ("time-series-regressor", "direct", DirectTimeSeriesRegressionForecaster),
    ("time-series-regressor", "recursive", RecursiveTimeSeriesRegressionForecaster),
    ("time-series-regressor", "multioutput", MultioutputTimeSeriesRegressionForecaster),
]


class _Recorder:
    # Helper class to record given data for later inspection.
    def fit(self, X, y):
        self.X_fit = X
        self.y_fit = y
        return self

    def predict(self, X):
        self.X_pred = X
        return np.ones(1)


class _TestTabularRegressor(BaseEstimator, RegressorMixin, _Recorder):
    pass


class _TestTimeSeriesRegressor(_Recorder, BaseRegressor):
    pass


@pytest.mark.parametrize(
    "estimator", [_TestTabularRegressor(), _TestTimeSeriesRegressor()]
)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("strategy", ["recursive", "direct", "multioutput"])
def test_consistent_data_passing_to_component_estimators_in_fit_and_predict(
    estimator, window_length, strategy
):
    # We generate data that represents time points in its values, i.e. an array of
    # values that increase in unit steps for each time point.
    n_variables = 3
    n_timepoints = 10
    y, X = _make_y_X(n_timepoints, n_variables)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=FH)

    forecaster = make_reduction(
        estimator, strategy=strategy, window_length=window_length
    )
    forecaster.fit(y_train, X_train, FH)
    forecaster.predict(X=X_test)

    # Get recorded data.
    if strategy == "direct":
        estimator_ = forecaster.estimators_[0]
    else:
        estimator_ = forecaster.estimator_

    X_fit = estimator_.X_fit
    y_fit = estimator_.y_fit
    X_pred = estimator_.X_pred

    # Format data into 3d array if the data is not in that format already.
    X_fit = X_fit.reshape(X_fit.shape[0], n_variables, -1)
    X_pred = X_pred.reshape(X_pred.shape[0], n_variables, -1)

    # Check that both fit and predict data have unit steps between them.
    assert np.allclose(np.diff(X_fit), 1)
    assert np.allclose(np.diff(X_pred), 1)

    # Check that predict data is a step ahead from last row in fit data.
    np.testing.assert_array_equal(X_pred, X_fit[[-1]] + 1)

    # Check that y values are further ahead than X values.
    assert np.all(X_fit < y_fit[:, np.newaxis, :])


@pytest.mark.parametrize("scitype, strategy, klass", _REGISTRY)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
def test_make_reduction_construct_instance(scitype, strategy, klass, window_length):
    estimator = DummyRegressor()
    forecaster = make_reduction(
        estimator, window_length=window_length, scitype=scitype, strategy=strategy
    )
    assert isinstance(forecaster, klass)
    assert forecaster.get_params()["window_length"] == window_length


@pytest.mark.parametrize(
    "estimator, scitype",
    [
        (LinearRegression(), "tabular-regressor"),
        (TimeSeriesForestRegressor(), "tabular-regressor"),
    ],
)
def test_make_reduction_infer_scitype(estimator, scitype):
    forecaster = make_reduction(estimator, scitype="infer")
    assert forecaster._estimator_scitype == scitype


def test_make_reduction_infer_scitype_raises_error():
    # The scitype of pipeline cannot be inferred here, as it may be used together
    # with a tabular or time series regressor.
    estimator = make_pipeline(Tabularizer(), LinearRegression())
    with pytest.raises(ValueError):
        make_reduction(estimator, scitype="infer")


@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_multioutput_direct_equivalence_tabular_linear_regression(fh):
    # multioutput and direct strategies with linear regression
    # regressor should produce same predictions
    y, X = make_forecasting_problem(make_X=True)
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)

    estimator = LinearRegression()
    direct = make_reduction(estimator, strategy="direct")
    multioutput = make_reduction(estimator, strategy="multioutput")

    y_pred_direct = direct.fit(y_train, X_train, fh=fh).predict(fh, X_test)
    y_pred_multioutput = multioutput.fit(y_train, X_train, fh=fh).predict(fh, X_test)

    np.testing.assert_array_almost_equal(
        y_pred_direct.to_numpy(), y_pred_multioutput.to_numpy()
    )
