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


@pytest.mark.skipif(
    not run_test_for_class(AutoEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["inverse-variance", "feature-importance"])
def test_autoensembler_hierarchical_global(method):
    """Test that global forecasters stay global in hierarchical data, see #8016.

    ``AutoEnsembleForecaster`` used to accept only ``Series`` data internally,
    so for ``Panel`` or ``Hierarchical`` data it was broadcast over instances,
    and a global component forecaster was fitted to each instance separately.
    """
    import numpy as np

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=30, max_timepoints=30, random_state=0
    )
    fh = [1, 2]

    global_forecaster = RecursiveTabularRegressionForecaster(
        LinearRegression(), window_length=3, pooling="global"
    )
    forecasters = [("global", global_forecaster), ("naive", NaiveForecaster())]

    ensemble_forecaster = AutoEnsembleForecaster(forecasters=forecasters, method=method)
    ensemble_forecaster.fit(y, fh=fh)

    # the ensemble is fitted once, to the entire hierarchical data,
    # and not broadcast over instances
    assert isinstance(ensemble_forecaster.forecasters_, list)
    assert len(ensemble_forecaster.weights_) == 2
    fitted_global = ensemble_forecaster.forecasters_[0][1]
    assert fitted_global._y.index.nlevels == 3
    assert fitted_global._y.index.droplevel(-1).nunique() == 4

    # predictions are a weighted average of the component predictions
    y_pred = ensemble_forecaster.predict()
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.index.nlevels == 3
    assert len(y_pred) == 4 * len(fh)
    assert list(y_pred.columns) == list(y.columns)

    predictions = pd.concat(
        [f.predict() for _, f in ensemble_forecaster.forecasters_], axis=1
    )
    expected = np.average(
        predictions.to_numpy(), weights=ensemble_forecaster.weights_, axis=1
    )
    np.testing.assert_allclose(y_pred.to_numpy().ravel(), expected)
