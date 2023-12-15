"""Test piecewise linear trend forecasters.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["sbuse"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import (
    PiecewiseLinearTrendForecaster,
    PolynomialTrendForecaster,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(PiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pred_errors_against_linear():
    """Check prediction performance on airline dataset.

    For a small value of changepoint_prior_scale like 0.001 the
    PiecewiseLinearTrendForecaster must return a linear trend.

    Arguments
    ---------
    fh: ForecastingHorizon, fh at which to test prediction

    Raises
    ------
    AssertionError - if forecast is not compatible with a linear trend.
    """
    y = load_airline().to_timestamp(freq="M")
    y_train, y_test = temporal_train_test_split(y)

    fh = ForecastingHorizon(y_test.index, is_relative=False)

    f = PiecewiseLinearTrendForecaster(changepoint_prior_scale=0.001)
    y_pred_f = f.fit(y_train).predict(fh)

    linear = PolynomialTrendForecaster(degree=1)
    y_pred_linear = linear.fit(y_train).predict(fh)

    np.testing.assert_allclose(y_pred_f, y_pred_linear, rtol=0.04)


@pytest.mark.skipif(
    not run_test_for_class(PiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pred_with_explicit_changepoints():
    """Check functionality with explicit changepoints.

    When changepoints are passed to the PiecewiseLinearTrendForecaster
    the prediction has to be different then the automatic detection because the
    changepoints are forcefully added.

    Raises
    ------
    AssertionError - if changepoints have no significant effect on the prediction.
    """
    y = load_airline().to_timestamp(freq="M")
    y_train, y_test = temporal_train_test_split(y)

    fh = ForecastingHorizon(y_test.index, is_relative=False)

    a = PiecewiseLinearTrendForecaster(changepoints=["1953-05-31"])
    b = PiecewiseLinearTrendForecaster()

    slope_a = a.fit(y_train).predict(fh).diff().mean()
    slope_b = b.fit(y_train).predict(fh).diff().mean()

    assert not np.allclose(slope_a, slope_b, rtol=0.1)


@pytest.mark.skipif(
    not run_test_for_class(PiecewiseLinearTrendForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("indextype", ["range", "period"])
def test_pwl_trend_nonnative_index(indextype):
    """Check pwl detrend with RangeIndex and PeriodIndex."""
    y = pd.DataFrame({"a": [1, 2, 3, 4]})

    if indextype == "period":
        y.index = pd.period_range("2000-01-01", periods=4)

    fh = [1, 2]

    f = PiecewiseLinearTrendForecaster()
    f.fit(y)
    y_pred = f.predict(fh=fh)

    if indextype == "range":
        assert pd.api.types.is_integer_dtype(y_pred.index)
    if indextype == "period":
        assert isinstance(y_pred.index, pd.PeriodIndex)
