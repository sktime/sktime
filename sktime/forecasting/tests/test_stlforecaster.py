#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Test for STLForecaster Module."""
__author__ = ["Taiwo Owoseni"]
__all__ = [
    "test_check_compare_stl_and_ttf_results",
    "test_check_compare_stl_and_ttf_results_naive",
]

import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.stlforecaster import STLForecaster
from sktime.forecasting.compose import TransformedTargetForecaster

naive_forecaster = NaiveForecaster(strategy="drift")
n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]
fh = [1, 3, 4]

estimator = ARIMA(order=(1, 1, 1))

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
ttf.fit(y_train)
ttf_naive.fit(y_train)

stlf = STLForecaster(
    estimator,
    Deseasonalizer(),
    Detrender(forecaster=PolynomialTrendForecaster(degree=1)),
)

stlf_naive = STLForecaster(
    naive_forecaster,
    Deseasonalizer(),
    Detrender(forecaster=PolynomialTrendForecaster(degree=1)),
)
stlf.fit(y_train)
stlf_naive.fit(y_train)


def test_check_compare_stl_and_ttf_results():
    """Compare two ARIMA Forecaster."""
    np.allclose(stlf.predict(fh), ttf.predict(fh))


def test_check_compare_stl_and_ttf_results_naive():
    """Compare two Naive Forecaster."""
    np.allclose(stlf_naive.predict(fh), ttf_naive.predict(fh))
