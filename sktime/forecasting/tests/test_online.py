#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["William Zheng"]

import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split

from sktime.forecasting.ensemble_algorithms import HedgeExpertEnsemble,\
                                                   NNLSEnsemble
from sktime.forecasting.online_experts import NormalHedge, se
from sktime.forecasting.online_ensemble import OnlineEnsembleForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def test_weights_for_airline_averaging():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = OnlineEnsembleForecaster([
        ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
        ("holt", ExponentialSmoothing(trend="add", damped=False,
                                      seasonal="multiplicative", sp=12)),
        ("damped", ExponentialSmoothing(trend="add", damped=True,
                                        seasonal="multiplicative", sp=12))
    ])

    forecaster.fit(y_train)

    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(forecaster.weights, expected, rtol=1e-8)


def test_weights_for_airline_normal_hedge():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    hedge_expert = HedgeExpertEnsemble(3, 80, NormalHedge, loss_func=se)

    forecaster = OnlineEnsembleForecaster([
        ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
        ("holt", ExponentialSmoothing(trend="add", damped=False,
                                      seasonal="multiplicative", sp=12)),
        ("damped", ExponentialSmoothing(trend="add", damped=True,
                                        seasonal="multiplicative", sp=12))
    ], ensemble_algorithm=hedge_expert)

    forecaster.fit(y_train)
    forecaster.update_predict(y_test)

    expected = np.array([0, 0.55132043, 0.44867957])
    np.testing.assert_allclose(forecaster.weights, expected, rtol=1e-8)


def test_weights_for_airline_nnls():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    hedge_expert = NNLSEnsemble(3, loss_func=se)

    forecaster = OnlineEnsembleForecaster([
        ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
        ("holt", ExponentialSmoothing(trend="add", damped=False,
                                      seasonal="multiplicative", sp=12)),
        ("damped", ExponentialSmoothing(trend="add", damped=True,
                                        seasonal="multiplicative", sp=12))
    ], ensemble_algorithm=hedge_expert)

    forecaster.fit(y_train)
    forecaster.update_predict(y_test)

    expected = np.array([0, 1.02354436, 0])
    np.testing.assert_allclose(forecaster.weights, expected, rtol=1e-8)
