#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test OnlineEnsembleForecaster."""

__author__ = ["magittan"]

import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.metrics import mean_squared_error

from sktime.datasets import load_airline
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning._online_ensemble import OnlineEnsembleForecaster
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NNLSEnsemble,
    NormalHedgeEnsemble,
)
from sktime.split import SlidingWindowSplitter, temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class

cv = SlidingWindowSplitter(start_with_window=True, window_length=1, fh=1)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_weights_for_airline_averaging():
    """Test weights."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = OnlineEnsembleForecaster(
        [
            ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
            (
                "holt",
                ExponentialSmoothing(
                    trend="add", damped_trend=False, seasonal="multiplicative", sp=12
                ),
            ),
            (
                "damped_trend",
                ExponentialSmoothing(
                    trend="add", damped_trend=True, seasonal="multiplicative", sp=12
                ),
            ),
        ]
    )

    forecaster.fit(y_train)

    expected = np.array([1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(forecaster.weights, expected, rtol=1e-8)


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_weights_for_airline_normal_hedge():
    """Test weights."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    hedge_expert = NormalHedgeEnsemble(n_estimators=3, loss_func=mean_squared_error)

    forecaster = OnlineEnsembleForecaster(
        [
            ("av5", NaiveForecaster(strategy="mean", window_length=5)),
            ("av10", NaiveForecaster(strategy="mean", window_length=10)),
            ("av20", NaiveForecaster(strategy="mean", window_length=20)),
        ],
        ensemble_algorithm=hedge_expert,
    )

    forecaster.fit(y_train)
    forecaster.update_predict(y=y_test, cv=cv, reset_forecaster=False)

    expected = np.array([0.17077154, 0.48156709, 0.34766137])
    np.testing.assert_allclose(forecaster.weights, expected, atol=1e-8)


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_weights_for_airline_nnls():
    """Test weights."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    hedge_expert = NNLSEnsemble(n_estimators=3, loss_func=mean_squared_error)

    forecaster = OnlineEnsembleForecaster(
        [
            ("av5", NaiveForecaster(strategy="mean", window_length=5)),
            ("av10", NaiveForecaster(strategy="mean", window_length=10)),
            ("av20", NaiveForecaster(strategy="mean", window_length=20)),
        ],
        ensemble_algorithm=hedge_expert,
    )

    forecaster.fit(y_train)
    forecaster.update_predict(y=y_test, cv=cv, reset_forecaster=False)

    expected = np.array([0.04720766, 0, 1.03410876])
    np.testing.assert_allclose(forecaster.weights, expected, atol=1e-8)


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ensemble_algorithm_param_not_mutated_and_reset_in_fit():
    """update must not change the ensemble_algorithm parameter, fit must reset it.

    The ensemble algorithm is stateful. The forecaster used to update the weights
    of the ensemble_algorithm object passed by the user, so that the parameter
    changed, and a refit continued from the weights of the previous updates.
    """
    y = load_airline()
    algorithm = NormalHedgeEnsemble(n_estimators=2, loss_func=mean_squared_error)
    initial_weights = algorithm.weights.copy()

    forecaster = OnlineEnsembleForecaster(
        [
            ("last", NaiveForecaster()),
            ("mean", NaiveForecaster(strategy="mean", window_length=12)),
        ],
        ensemble_algorithm=algorithm,
    )
    forecaster.fit(y[:100], fh=[1, 2])
    forecaster.update(y[100:120])
    weights_after_update = forecaster.weights.copy()

    # the weights were updated, but not those of the parameter
    assert not np.allclose(weights_after_update, initial_weights)
    np.testing.assert_array_equal(algorithm.weights, initial_weights)
    assert forecaster.ensemble_algorithm is algorithm

    # refitting starts again from the initial weights
    forecaster.fit(y[:100], fh=[1, 2])
    np.testing.assert_allclose(forecaster.weights, initial_weights)

    # and repeating the updates gives the same weights as before
    forecaster.update(y[100:120])
    np.testing.assert_allclose(forecaster.weights, weights_after_update)


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_does_not_change_weights():
    """predict must not write to the forecaster."""
    y = load_airline()
    forecaster = OnlineEnsembleForecaster(
        [("last", NaiveForecaster()), ("drift", NaiveForecaster(strategy="drift"))],
        ensemble_algorithm=NormalHedgeEnsemble(
            n_estimators=2, loss_func=mean_squared_error
        ),
    )
    forecaster.fit(y[:100], fh=[1, 2])
    forecaster.update(y[100:110])
    weights = forecaster.weights.copy()

    forecaster.predict()
    np.testing.assert_array_equal(forecaster.weights, weights)
