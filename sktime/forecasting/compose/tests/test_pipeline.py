#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd

from sktime.datasets import load_airline
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.outlier_detection import HampelFilter


def test_pipeline():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = TransformedTargetForecaster(
        [
            ("t1", Deseasonalizer(sp=12, model="multiplicative")),
            ("t2", Detrender(PolynomialTrendForecaster(degree=1))),
            ("forecaster", NaiveForecaster()),
        ]
    )
    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh)
    actual = forecaster.predict()

    def compute_expected_y_pred(y_train, fh):
        # fitting
        yt = y_train.copy()
        t1 = Deseasonalizer(sp=12, model="multiplicative")
        yt = t1.fit_transform(yt)
        t2 = Detrender(PolynomialTrendForecaster(degree=1))
        yt = t2.fit_transform(yt)
        forecaster = NaiveForecaster()
        forecaster.fit(yt, fh=fh)

        # predicting
        y_pred = forecaster.predict()
        y_pred = t2.inverse_transform(y_pred)
        y_pred = t1.inverse_transform(y_pred)
        return y_pred

    expected = compute_expected_y_pred(y_train, fh)
    np.testing.assert_array_equal(actual, expected)


def test_skip_inverse_transform():
    # testing that transformers which have the "skip-inverse-transform" tag
    # are working in a pipeline
    y = load_airline()
    # add nan and outlier
    y.iloc[3] = np.nan
    y.iloc[4] = y.iloc[4] * 20

    y_train, y_test = temporal_train_test_split(y)
    forecaster = TransformedTargetForecaster(
        [
            ("t1", HampelFilter(window_length=12)),
            ("t2", Imputer(method="mean")),
            ("forecaster", NaiveForecaster()),
        ]
    )
    fh = np.arange(len(y_test)) + 1
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict()
    assert isinstance(y_pred, pd.Series)
