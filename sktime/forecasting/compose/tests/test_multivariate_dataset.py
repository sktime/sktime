#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Sebastiaan Koel"]
__all__ = []

import numpy as np
from sktime.datasets import load_uschange
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.detrend import Deseasonalizer
from sktime.transformers.single_series.detrend import Detrender


def test_pipeline():
    X, y = load_uschange()

    # split the y data into a train and test sample
    # use the y index to select the matching X sample
    y_train, y_test = temporal_train_test_split(y)
    X_train, X_test = X.iloc[y_train.index,:], X.iloc[y_test.index, :]

    # Todo NaiveForecaster() ignores X_train
    # so need to replace with a forecaster which
    # uses X_train to implement _transform() in _reduce.py

    f = TransformedTargetForecaster([
        ("f", NaiveForecaster())
    ])
    fh = np.arange(len(y_test)) + 1
    f.fit(y_train, fh, X_train=X_train)
    
    
    # actual = f.predict()

    # def compute_expected_y_pred(y_train, fh):
    #     # fitting
    #     yt = y_train.copy()
    #     t1 = Deseasonalizer(sp=12, model="multiplicative")
    #     yt = t1.fit_transform(yt)
    #     t2 = Detrender(PolynomialTrendForecaster(degree=1))
    #     yt = t2.fit_transform(yt)
    #     f = NaiveForecaster()
    #     f.fit(yt, fh)

    #     # predicting
    #     y_pred = f.predict()
    #     y_pred = t2.inverse_transform(y_pred)
    #     y_pred = t1.inverse_transform(y_pred)
    #     return y_pred

    # expected = compute_expected_y_pred(y_train, fh)
    # np.testing.assert_array_equal(actual, expected)