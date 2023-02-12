#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test OnlineEnsembleForecaster."""

__author__ = ["magittan"]

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from sktime.datasets import load_airline
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.online_learning._online_ensemble import OnlineEnsembleForecaster
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NNLSEnsemble,
    NormalHedgeEnsemble,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

cv = SlidingWindowSplitter(start_with_window=True, window_length=1, fh=1)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
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
