#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel"]
__all__ = []

import numpy as np
from sktime.datasets.base import load_uschange
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import ReducedRegressionForecaster


def test_multivariate():
    X, y = load_uschange()

    y_train, y_test = temporal_train_test_split(y)
    X_train = X.iloc[y_train.index, :]

    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    window_length = 2

    regressor = LinearRegression()
    forecaster = ReducedRegressionForecaster(regressor, window_length=window_length)
    forecaster.fit(y_train, fh=fh, X_train=X_train)

    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(y_test)

    coefs = forecaster.regressor_.coef_
    intercept = forecaster.regressor_.intercept_

    assert len(coefs) == (len(X.columns) + 1) * window_length
    assert isinstance(intercept, float)
    assert 0.33 < coefs[0] < 0.34
    assert -0.09 > coefs[-1] > -0.10
    assert 0.39 < intercept < 0.40

    forecaster = ReducedRegressionForecaster(
        regressor, strategy="direct", window_length=window_length
    )

    forecaster.fit(y_train, fh=fh, X_train=X_train)
    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(y_test)

    regressors = forecaster.regressors_

    assert len(regressors) == len(y_test)
    assert len(regressors[0].coef_) == (len(X.columns) + 1) * window_length
    assert isinstance(regressors[0].intercept_, float)

    assert 0.75 < regressors[0].coef_[0] < 0.76
    assert -0.20 > regressors[0].coef_[-1] > -0.21
    assert 0.20 < regressors[0].intercept_ < 0.21

    assert -0.02 > regressors[-1].coef_[0] > -0.03
    assert 0.40 < regressors[-1].coef_[-1] < 0.41
    assert 0.77 < regressors[-1].intercept_ < 0.79
