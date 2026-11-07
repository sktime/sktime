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
@pytest.mark.parametrize("n_estimators", [1, 2, 3])
def test_normal_hedge_zero_regret(n_estimators):
    """Test NormalHedgeEnsemble keeps uniform weights if no regret is positive.

    Failure case: identical predictions give zero regret, which used to cause
    a division by zero and a ValueError from the root finder.
    """
    hedge_expert = NormalHedgeEnsemble(
        n_estimators=n_estimators, loss_func=mean_squared_error
    )
    y_pred = np.tile([1.0, 2.0, 3.0], (n_estimators, 1))
    y_true = np.array([1.5, 2.5, 3.5])

    hedge_expert.update(y_pred, y_true)

    expected = np.ones(n_estimators) / n_estimators
    np.testing.assert_allclose(hedge_expert.weights, expected)


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_normal_hedge_recovers_after_zero_regret():
    """Test NormalHedgeEnsemble weights still adapt after a zero regret step."""
    hedge_expert = NormalHedgeEnsemble(n_estimators=2, loss_func=mean_squared_error)

    # equal losses: no positive regret, weights stay uniform
    hedge_expert.update(np.array([[1.0], [1.0]]), np.array([2.0]))
    np.testing.assert_allclose(hedge_expert.weights, [0.5, 0.5])

    # first estimator is exact, so it gets all the weight
    hedge_expert.update(np.array([[2.0], [5.0]]), np.array([2.0]))
    np.testing.assert_allclose(hedge_expert.weights, [1.0, 0.0])


@pytest.mark.skipif(
    not run_test_for_class(OnlineEnsembleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_online_ensemble_normal_hedge_identical_forecasters():
    """Test OnlineEnsembleForecaster with NormalHedge and identical forecasters."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    hedge_expert = NormalHedgeEnsemble(n_estimators=2, loss_func=mean_squared_error)

    forecaster = OnlineEnsembleForecaster(
        [
            ("naive1", NaiveForecaster()),
            ("naive2", NaiveForecaster()),
        ],
        ensemble_algorithm=hedge_expert,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.update_predict(y=y_test, cv=cv, reset_forecaster=False)

    assert not y_pred.isna().any()
    np.testing.assert_allclose(forecaster.weights, [0.5, 0.5])
