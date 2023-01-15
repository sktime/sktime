# -*- coding: utf-8 -*-
"""Test detrenders."""
import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.tests.test_trend import get_expected_polynomial_coefs
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Detrender

__author__ = ["mloning", "KishManani"]
__all__ = []


@pytest.fixture()
def y_series():
    return load_airline()


@pytest.fixture()
def y_dataframe():
    return load_airline().to_frame()


def test_polynomial_detrending():
    """Test that transformer results agree with manual detrending."""
    y = pd.Series(np.arange(20) * 0.5) + np.random.normal(0, 1, size=20)
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check coefficients
    actual_coefs = transformer.forecaster_.regressor_.steps[-1][-1].coef_
    expected_coefs = get_expected_polynomial_coefs(y, degree=1, with_intercept=True)[
        ::-1
    ]
    np.testing.assert_array_almost_equal(actual_coefs, expected_coefs)

    # check trend
    n = len(y)
    expected_trend = expected_coefs[0] + np.arange(n) * expected_coefs[1]
    expected_trend_2D = np.reshape(expected_trend, (n, 1))
    actual_trend = transformer.forecaster_.predict(-np.arange(n))
    np.testing.assert_array_almost_equal(actual_trend, expected_trend_2D)

    # check residuals
    actual = transformer.transform(y)
    expected = y - expected_trend
    np.testing.assert_array_almost_equal(actual, expected)


def test_multiplicative_detrending_series(y_series):
    """Tests we get the expected result when setting `model=multiplicative`."""
    # Load test dataset
    y = y_series

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="multiplicative")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y / trend

    pd.testing.assert_series_equal(y_transformed, expected)


def test_multiplicative_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=multiplicative`."""
    # Load test dataset
    y = y_dataframe

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="multiplicative")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y / trend

    pd.testing.assert_frame_equal(y_transformed, expected)


def test_additive_detrending_series(y_series):
    """Tests we get the expected result when setting `model=additive`."""
    # Load test dataset
    y = y_series

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="additive")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y - trend

    pd.testing.assert_series_equal(y_transformed, expected)


def test_additive_detrending_dataframe(y_dataframe):
    """Tests we get the expected result when setting `model=additive`."""
    # Load test dataset
    y = y_dataframe

    # Get the trend
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    trend = forecaster.fit_predict(y, fh=y.index)

    # De-trend the time series
    detrender = Detrender(forecaster, model="additive")
    y_transformed = detrender.fit_transform(y)

    # Compute the expected de-trended time series
    expected = y - trend

    pd.testing.assert_frame_equal(y_transformed, expected)
