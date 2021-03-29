#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal"]
__all__ = []

import numpy as np

import pytest

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.panel.reduce import Tabularizer
from sktime.forecasting.compose._reduce import ReducedForecaster
from sktime.forecasting.compose._reduce import RecursiveRegressionForecaster
from sktime.forecasting.compose._reduce import DirectRegressionForecaster
from sktime.forecasting.compose._reduce import MultioutputRegressionForecaster
from sktime.forecasting.compose._reduce import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import DirectTimeSeriesRegressionForecaster

# precision for assert_almost_equal tests
# chosen by Lovkush Agarwal. No particular reason for 5. Not too big or too small
DECIMAL = 5

# this expected value created by Lovkush Agarwal by running code locally
EXPECTED_AIRLINE_LINEAR_RECURSIVE = [
    397.28122475088117,
    391.0055770755232,
    382.85931770491493,
    376.75382498759643,
    421.3439733242519,
    483.7127665080476,
    506.5011555360703,
    485.95155173523494,
    414.41328025499604,
    371.2843322707713,
    379.5680077722808,
    406.146827316167,
    426.48249271837176,
    415.5337957767289,
    405.48715913377714,
    423.97150765765025,
    472.10998764155966,
    517.7763038626333,
    515.6077989417864,
    475.8615207069196,
    432.47049089698646,
    417.62468250043514,
    435.3174101071012,
    453.8693707695759,
]

# this expected value created by Lovkush Agarwal by running code locally
EXPECTED_AIRLINE_LINEAR_DIRECT = [
    388.7894742436609,
    385.4311737990922,
    404.66760376792183,
    389.3921653574014,
    413.5415037170552,
    491.27471550855756,
    560.5985060880608,
    564.1354313250545,
    462.8049467298484,
    396.8247623180332,
    352.5416937680942,
    369.3915756974357,
    430.12889943026323,
    417.13419789042484,
    434.8091175980315,
    415.33997516059355,
    446.97711875155846,
    539.6761098618977,
    619.7204673400846,
    624.3153932803112,
    499.686252475341,
    422.0658526180952,
    373.3847171492921,
    388.8020135264563,
]


@pytest.mark.parametrize(
    "forecaster, expected",
    [
        (
            RecursiveRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            ReducedForecaster(
                LinearRegression(), scitype="regressor", strategy="recursive"
            ),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            DirectRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            ReducedForecaster(
                LinearRegression(), scitype="regressor", strategy="direct"
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            RecursiveTimeSeriesRegressionForecaster(
                Pipeline([("tabularize", Tabularizer()), ("model", LinearRegression())])
            ),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            ReducedForecaster(
                Pipeline(
                    [("tabularize", Tabularizer()), ("model", LinearRegression())]
                ),
                scitype="ts_regressor",
                strategy="recursive",
            ),
            EXPECTED_AIRLINE_LINEAR_RECURSIVE,
        ),
        (
            DirectTimeSeriesRegressionForecaster(
                Pipeline([("tabularize", Tabularizer()), ("model", LinearRegression())])
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            ReducedForecaster(
                Pipeline(
                    [("tabularize", Tabularizer()), ("model", LinearRegression())]
                ),
                scitype="ts_regressor",
                strategy="direct",
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            MultioutputRegressionForecaster(LinearRegression()),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
        (
            ReducedForecaster(
                LinearRegression(), scitype="regressor", strategy="multioutput"
            ),
            EXPECTED_AIRLINE_LINEAR_DIRECT,
        ),
    ],
)
def test_reductions_airline_data(forecaster, expected):
    """
    test the various reduction forecasters by
    computing predictions on airline data and comparing to values
    calculated by Lovkush Agarwal on their local machine in Mar 2021
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    actual = forecaster.fit(y_train, fh=fh).predict(fh)

    np.testing.assert_almost_equal(actual, expected, decimal=DECIMAL)
