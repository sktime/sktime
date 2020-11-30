#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel"]
__all__ = []

import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import ReducedRegressionForecaster

from sklearn.utils._testing import assert_array_almost_equal


# Generate data to know the outcome of the regressors
L = 30
X = [[x for x in range(L)], [x + 1 for x in range(L)]]
y = [1, 2]
W = [0.5, 0.5, 0.1, 0.1, 0.1, 0.1]

for i in range(L - 2):
    yi = (
        W[0] * y[i + 1]
        + W[1] * y[i]
        + W[2] * X[0][i + 1]
        + W[3] * X[0][i]
        + W[4] * X[1][i + 1]
        + W[5] * X[1][i]
    )
    y.append(yi)

X = pd.DataFrame(X, copy=True).T
y = pd.Series(y)


def test_multivariate_recursive():

    y_train, y_test = temporal_train_test_split(y)
    X_train = X.iloc[y_train.index, :]
    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    window_length = 2

    regressor = LinearRegression(fit_intercept=False)
    forecaster = ReducedRegressionForecaster(regressor, window_length=window_length)

    forecaster.fit(y_train, fh=fh, X_train=X_train)
    coefs = forecaster.regressor_.coef_
    intercept = forecaster.regressor_.intercept_

    y_pred = forecaster.predict(fh=fh)

    assert_array_almost_equal(coefs, W)
    assert_array_almost_equal(intercept, [0])
    assert_array_almost_equal(y_pred.iloc[0], y_test.iloc[0])

    assert len(coefs) == (len(X.columns) + 1) * window_length


# test_multivariate_recursive()


def test_multivariate_direct():

    y_train, y_test = temporal_train_test_split(y)
    X_train = X.iloc[y_train.index, :]

    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    window_length = 2

    regressor = LinearRegression(fit_intercept=False)
    forecaster = ReducedRegressionForecaster(
        regressor, strategy="direct", window_length=window_length
    )

    forecaster.fit(y_train, fh=fh, X_train=X_train)
    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(y_test)

    regressors = forecaster.regressors_

    assert len(regressors) == len(y_test)
    assert_array_almost_equal(regressors[0].coef_, W)
    assert_array_almost_equal(y_test, y_pred)


test_multivariate_direct()
