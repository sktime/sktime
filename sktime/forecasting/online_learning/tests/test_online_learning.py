#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["William Zheng"]

import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
    NormalHedgeEnsemble,
    NNLSEnsemble,
)
from sktime.forecasting.online_learning._online_ensemble import (
    OnlineEnsembleForecaster,
)

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sklearn.metrics import mean_squared_error


def test_weights_for_airline_averaging():
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
    forecaster.update_predict(y_test)

    expected = np.array([0.17077154, 0.48156709, 0.34766137])
    np.testing.assert_allclose(forecaster.weights, expected, atol=1e-8)


def test_weights_for_airline_nnls():
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
    forecaster.update_predict(y_test)

    expected = np.array([0.04720766, 0, 1.03410876])
    np.testing.assert_allclose(forecaster.weights, expected, atol=1e-8)
