#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

"""Test for STLForecaster Module"""
__author__ = ["Taiwo Owoseni"]
__all__ = ["check_compare_stl_and_ttf_results"]

import numpy as np
import pandas as pd
from sktime.forecasting.arima import ARIMA
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.stlforecaster import STLForecaster
from sktime.forecasting.compose import TransformedTargetForecaster

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]

estimator = ARIMA()
stl_forecaster_steps = [
    ("deseasonalise", Deseasonalizer()),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
]
transformed_target_forecaster_steps = [
    ("deseasonalise", Deseasonalizer()),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("estimator", estimator),
]
ttf = TransformedTargetForecaster(transformed_target_forecaster_steps)
ttf.fit(y_train)

stlf = STLForecaster(estimator, stl_forecaster_steps)
stlf.fit(y_train)


def check_compare_stl_and_ttf_results():
    """Compare two Forecaster."""
    np.testing.assert_allclose(stlf.predict(fh=[1, 3, 4]), ttf.predict(fh=[1, 3, 4]))
