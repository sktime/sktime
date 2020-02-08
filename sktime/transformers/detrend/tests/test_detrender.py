#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

from sktime.transformers.detrend import Detrender
from sktime.forecasting.reduction import ReducedRegressionForecaster
from sklearn.linear_model import LinearRegression
from sktime.forecasting.model_selection import SlidingWindowSplitter
import numpy as np
import pandas as pd


n = 100
trend = np.arange(n)
noise = np.random.normal(loc=0, scale=5, size=n)
data = trend + noise
y = pd.Series(data)
y_train = y.iloc[:90]
y_test = y.iloc[90:]


def test_detrender_fit_transform():
    forecaster = ReducedRegressionForecaster(regressor=LinearRegression(), cv=SlidingWindowSplitter())
    d = Detrender(forecaster=forecaster)
    d.fit(y_train)
    yt = d.transform(y_train)

    # compare time indices
    np.testing.assert_array_equal(yt.index.values, y_train.index.values)




