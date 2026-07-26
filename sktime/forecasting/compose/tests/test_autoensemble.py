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
