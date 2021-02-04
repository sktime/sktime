#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal"]
__all__ = []

import numpy as np

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.compose._reduce import ReducedForecaster
from sktime.forecasting.compose._reduce import RecursiveRegressionForecaster
from sktime.forecasting.compose._reduce import DirectRegressionForecaster
from sklearn.linear_model import LinearRegression


def test_factory_method_recursive():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="recursive")
    f2 = RecursiveRegressionForecaster(regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = f2.fit(y_train).predict(fh)

    np.testing.assert_array_equal(actual, expected)


def test_factory_method_direct():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="direct")
    f2 = DirectRegressionForecaster(regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = f2.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_array_equal(actual, expected)
