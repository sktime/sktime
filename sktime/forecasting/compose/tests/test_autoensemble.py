#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of AutoEnsembleForecaster functionality."""

__author__ = ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"]

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sktime.datasets import load_longley
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    AutoEnsembleForecaster,
    RecursiveTabularRegressionForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(AutoEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecasters",
    [
        [
            (
                "dt",
                RecursiveTabularRegressionForecaster(
                    DecisionTreeRegressor(random_state=42), window_length=3
                ),
            ),
            (
                "lr",
                RecursiveTabularRegressionForecaster(
                    LinearRegression(), window_length=3
                ),
            ),
        ],
    ],
)
@pytest.mark.parametrize(
    "method",
    ["inverse-variance", "feature-importance"],
)
def test_autoensembler(forecasters, method):
    """Check that the prediction is a weighted mean of the individual predictions."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    fh_test = ForecastingHorizon(y_test.index, is_relative=False)

    ensemble_forecaster = AutoEnsembleForecaster(forecasters=forecasters, method=method)
    ensemble_forecaster.fit(y_train, X_train)
    y_pred = ensemble_forecaster.predict(fh=fh_test, X=X_test)

    predictions = []
    for _, forecaster in forecasters:
        f = forecaster
        f.fit(y_train, X_train)
        f_pred = f.predict(fh=fh_test, X=X_test)
        predictions.append(f_pred)
    predictions = pd.DataFrame(predictions).T

    assert (predictions.min(axis=1) <= y_pred).all()
    assert (predictions.max(axis=1) >= y_pred).all()


@pytest.mark.skipif(
    not run_test_for_class(AutoEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_autoensembler_inverse_variance_zero_variance():
    """Weights/predictions must not be NaN if a forecaster fits the test set exactly.

    See bug report in #4212. A zero test-set variance for one or more
    forecasters previously produced NaN weights via a 1/0 division.
    """
    y = pd.Series(
        [
            8.0,
            8.0,
            9.0,
            9.0,
            9.0,
            9.0,
            8.0,
            8.0,
            9.0,
            8.0,
            9.0,
            10.0,
            9.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            11.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]
    )
    forecasters = [
        ("naive", NaiveForecaster(strategy="last")),
        ("exponential", ExponentialSmoothing()),
    ]

    forecaster = AutoEnsembleForecaster(
        forecasters=forecasters, method="inverse-variance"
    )
    forecaster.fit(y, fh=list(range(1, 13)))
    y_pred = forecaster.predict()

    assert not any(pd.isna(w) for w in forecaster.weights_)
    assert abs(sum(forecaster.weights_) - 1.0) < 1e-9
    assert not y_pred.isna().any()
