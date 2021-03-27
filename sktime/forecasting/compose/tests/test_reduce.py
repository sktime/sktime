#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal"]
__all__ = []

import numpy as np

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


def test_recursive():
    """
    testing the RecursiveRegressionForecaster
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = RecursiveRegressionForecaster(regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_factory_method_recursive():
    """
    testing the factory method agrees with RecursiveRegressionForecaster
    `expected` here is the same as `expected` in the test
    directly above
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="recursive")

    actual = f1.fit(y_train).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_direct():
    """
    testing the DirectRegressionForecaster
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = DirectRegressionForecaster(regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_factory_method_direct():
    """
    testing the factory method agrees with DirectRegressionForecaster
    `expected` here is the same as `expected` in the test
    directly above
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = ReducedForecaster(regressor, scitype="regressor", strategy="direct")

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_ts_recursive():
    """
    testing the RecursiveTimeSeriesRegressionForecaster
    note that `expected` here matches `expected`` from tabular recursive test
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = RecursiveTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_factory_method_ts_recursive():
    """
    testing the factory method agrees with RecursiveTimeSeriesRegressionForecaster
    `expected` here is the same as `expected` in the test
    directly above
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = ReducedForecaster(ts_regressor, scitype="ts_regressor", strategy="recursive")

    actual = f1.fit(y_train).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_ts_direct():
    """
    testing the DirectTimeSeriesRegressionForecaster
    note that `expected` here matches `expected` from tabular direct test
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = DirectTimeSeriesRegressionForecaster(ts_regressor)

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_factory_method_ts_direct():
    """
    testing the factory method agrees with DirectTimeSeriesRegressionForecaster
    `expected` here is the same as `expected` in the test
    directly above
    """
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    ts_regressor = Pipeline(
        [("tabularize", Tabularizer()), ("model", LinearRegression())]
    )
    f1 = ReducedForecaster(ts_regressor, scitype="ts_regressor", strategy="direct")

    actual = f1.fit(y_train, fh=fh).predict(fh)
    expected = [
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

    np.testing.assert_almost_equal(actual, expected, decimal=5)


def test_multioutput_direct_tabular():
    # multioutput and direct strategies with linear regression
    # regressor should produce same predictions
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=24)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = LinearRegression()
    f1 = MultioutputRegressionForecaster(regressor)
    f2 = DirectRegressionForecaster(regressor)

    preds1 = f1.fit(y_train, fh=fh).predict(fh)
    preds2 = f2.fit(y_train, fh=fh).predict(fh)

    # assert_almost_equal does not seem to work with pd.Series objects
    np.testing.assert_almost_equal(preds1.to_numpy(), preds2.to_numpy(), decimal=5)
