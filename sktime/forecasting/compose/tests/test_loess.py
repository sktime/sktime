#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Test for STLForecaster Module."""
__author__ = ["Taiwo Owoseni"]
__all__ = [
    "test_check_compare_stl_and_ttf_results",
    "test_check_compare_stl_and_ttf_results_naive",
]

import numpy as np

from sktime.forecasting.arima import ARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.compose._loess import STLForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = np.arange(len(y_test)) + 1

estimator = ARIMA(order=(1, 1, 1))
naive_forecaster = NaiveForecaster(strategy="drift")
transformed_target_forecaster_steps = [
    ("deseasonalise", Deseasonalizer()),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("estimator", estimator),
]
transformed_target_forecaster_steps_naive = [
    ("deseasonalise", Deseasonalizer()),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("estimator", naive_forecaster),
]
ttf = TransformedTargetForecaster(transformed_target_forecaster_steps)
ttf_naive = TransformedTargetForecaster(transformed_target_forecaster_steps_naive)
ttf.fit(y_train, fh=fh)
ttf_naive.fit(y_train, fh=fh)

stlf = STLForecaster(estimator, 1, 1)

stlf_naive = STLForecaster(naive_forecaster, 1, 1)
stlf.fit(y_train, fh=fh)
stlf_naive.fit(y_train, fh=fh)


def test_check_compare_stl_and_ttf_results():
    """Compare two ARIMA Forecaster."""
    np.allclose(stlf.predict(), ttf.predict())


def test_check_compare_stl_and_ttf_results_naive():
    """Compare two Naive Forecaster."""
    np.allclose(stlf_naive.predict(), ttf_naive.predict())
