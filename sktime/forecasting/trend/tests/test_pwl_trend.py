"""Test piecewise linear trend forecasters.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["sbuse"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import (
    PolynomialTrendForecaster,
    ProphetPiecewiseLinearTrendForecaster,
)
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ProphetPiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_for_changes_in_original():
    """Check if the original prophet implementation returns the same result as the
    sktime wrapper for the airline dataset.

    Raises
    ------
    AssertionError - if the predictions are not exactly the same
    """
    from prophet import Prophet

    from sktime.forecasting.fbprophet import Prophet as skProphet

    y = load_airline().to_timestamp(freq="M")

    # ------original Prophet---------
    prophet = Prophet()
    prophet.fit(pd.DataFrame(data={"ds": y.index, "y": y.values}))
    future = prophet.make_future_dataframe(periods=12, freq="M", include_history=False)
    forecast = prophet.predict(future)[["ds", "yhat"]]
    y_pred_original = forecast["yhat"]
    y_pred_original.index = forecast["ds"].values

    # ------sktime Prophet-----------
    skprophet = skProphet()
    y_pred_sktime = skprophet.fit_predict(y, fh=np.arange(1, 13))

    np.testing.assert_array_equal(y_pred_original.values, y_pred_sktime.values)  # exact


@pytest.mark.skipif(
    not run_test_for_class(ProphetPiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pred_errors_against_linear():
    """Check prediction performance on airline dataset.

    For a small value of changepoint_prior_scale like 0.001 the
    ProphetPiecewiseLinearTrendForecaster must return a single straight trendline.

    Raises
    ------
    AssertionError - if the trend forecast is not compatible with a linear trend.
    """
    y = load_airline().to_timestamp(freq="M")
    fh = ForecastingHorizon(y.index, is_relative=False)

    pwl = ProphetPiecewiseLinearTrendForecaster(changepoint_prior_scale=0.001)
    y_pred_pwl = pwl.fit(y).predict(fh)

    linear = PolynomialTrendForecaster(degree=1)
    y_pred_linear = linear.fit(y).predict(fh)

    np.testing.assert_allclose(y_pred_pwl, y_pred_linear, rtol=0.04)


@pytest.mark.skipif(
    not run_test_for_class(ProphetPiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pred_with_explicit_changepoints():
    """Check functionality with explicit changepoints.

    When changepoints are passed to the ProphetPiecewiseLinearTrendForecaster
    the prediction has to be different then the automatic detection because the
    changepoints are forcefully added.

    Raises
    ------
    AssertionError - if adding a changepoint has no effect on the trend prediction.
    """
    y = load_airline().to_timestamp(freq="M")
    y_train, y_test = temporal_train_test_split(y)

    fh = ForecastingHorizon(y_test.index, is_relative=False)
    seasonality_params = {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": True,
    }
    a = ProphetPiecewiseLinearTrendForecaster(
        changepoints=["1953-05-31"], **seasonality_params
    )
    b = ProphetPiecewiseLinearTrendForecaster(**seasonality_params)

    slope_a = a.fit(y_train).predict(fh).diff().mean()
    slope_b = b.fit(y_train).predict(fh).diff().mean()

    assert not np.allclose(slope_a, slope_b, rtol=0.1)


@pytest.mark.skipif(
    not run_test_for_class(ProphetPiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("indextype", ["range", "period"])
def test_pwl_trend_nonnative_index(indextype):
    """Check pwl detrend with RangeIndex and PeriodIndex."""
    y = pd.DataFrame({"a": [1, 2, 3, 4]})

    if indextype == "period":
        y.index = pd.period_range("2000-01-01", periods=4)

    fh = [1, 2]

    f = ProphetPiecewiseLinearTrendForecaster()
    f.fit(y)
    y_pred = f.predict(fh=fh)

    if indextype == "range":
        assert pd.api.types.is_integer_dtype(y_pred.index)
    if indextype == "period":
        assert isinstance(y_pred.index, pd.PeriodIndex)
