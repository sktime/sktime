#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# test API provided through BaseSktimeForecaster

__author__ = ["@mloning"]
__all__ = [
    "test_different_fh_in_fit_and_predict_req",
    "test_fh_in_fit_opt",
    "test_fh_in_fit_req",
    "test_fh_in_predict_opt",
    "test_no_fh_in_fit_req",
    "test_no_fh_opt",
    "test_oh_setting",
    "test_same_fh_in_fit_and_predict_opt",
    "test_same_fh_in_fit_and_predict_req",
]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import all_estimators
from sktime.utils._testing.estimator_checks import _construct_instance
from sktime.utils._testing.forecasting import make_forecasting_problem

# get all forecasters
FORECASTERS = [
    forecaster
    for (name, forecaster) in all_estimators(estimator_types="forecaster")
    if issubclass(forecaster, BaseForecaster)
]
FH0 = 1

WINDOW_FORECASTERS = [
    forecaster
    for (name, forecaster) in all_estimators(estimator_types="forecaster")
    if issubclass(forecaster, _BaseWindowForecaster)
]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


# test _y setting
@pytest.mark.parametrize("Forecaster", FORECASTERS)
def test_oh_setting(Forecaster):
    # check _y and cutoff is None after construction
    f = _construct_instance(Forecaster)
    assert f._y is None
    assert f.cutoff is None

    # check that _y and cutoff is updated during fit
    f.fit(y_train, fh=FH0)
    assert isinstance(f._y, pd.Series)
    assert len(f._y) > 0
    assert f.cutoff == y_train.index[-1]

    # check data pointers
    np.testing.assert_array_equal(f._y.index, y_train.index)

    # check that _y and cutoff is updated during update
    f.update(y_test, update_params=False)
    np.testing.assert_array_equal(f._y.index, np.append(y_train.index, y_test.index))
    assert f.cutoff == y_test.index[-1]


# check setting/getting API for forecasting horizon

# divide Forecasters into groups based on when fh is required
FORECASTERS_REQUIRED = [f for f in FORECASTERS if f._all_tags()["requires-fh-in-fit"]]

FORECASTERS_OPTIONAL = [
    f for f in FORECASTERS if not f._all_tags()["requires-fh-in-fit"]
]


# testing Forecasters which require fh during fitting
@pytest.mark.parametrize("Forecaster", FORECASTERS_REQUIRED)
def test_no_fh_in_fit_req(Forecaster):
    f = _construct_instance(Forecaster)
    # fh required in fit, raises error if not passed
    with pytest.raises(ValueError):
        f.fit(y_train)


@pytest.mark.parametrize("Forecaster", FORECASTERS_REQUIRED)
def test_fh_in_fit_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=FH0)
    np.testing.assert_array_equal(f.fh, FH0)
    f.predict()
    np.testing.assert_array_equal(f.fh, FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS_REQUIRED)
def test_same_fh_in_fit_and_predict_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=FH0)
    np.testing.assert_array_equal(f.fh, FH0)
    f.predict(FH0)
    np.testing.assert_array_equal(f.fh, FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS_REQUIRED)
def test_different_fh_in_fit_and_predict_req(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=FH0)
    np.testing.assert_array_equal(f.fh, FH0)
    # updating fh during predict raises error as fitted model depends on fh
    # seen in fit
    with pytest.raises(ValueError):
        f.predict(fh=FH0 + 1)


# testing Forecasters which take fh either during fitting or predicting
@pytest.mark.parametrize("Forecaster", FORECASTERS_OPTIONAL)
def test_no_fh_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train)
    # not passing fh to either fit or predict raises error
    with pytest.raises(ValueError):
        f.predict()


@pytest.mark.parametrize("Forecaster", FORECASTERS_OPTIONAL)
def test_fh_in_fit_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=FH0)
    np.testing.assert_array_equal(f.fh, FH0)
    f.predict()
    np.testing.assert_array_equal(f.fh, FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS_OPTIONAL)
def test_fh_in_predict_opt(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train)
    f.predict(FH0)
    np.testing.assert_array_equal(f.fh, FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS_OPTIONAL)
def test_same_fh_in_fit_and_predict_opt(Forecaster):
    f = _construct_instance(Forecaster)
    # passing the same fh to both fit and predict works
    f.fit(y_train, fh=FH0)
    f.predict(FH0)
    np.testing.assert_array_equal(f.fh, FH0)


@pytest.mark.parametrize("Forecaster", WINDOW_FORECASTERS)
def test_last_window(Forecaster):
    f = _construct_instance(Forecaster)
    # passing the same fh to both fit and predict works
    f.fit(y_train, fh=FH0)

    actual, _ = f._get_last_window()
    expected = y_train.iloc[-f.window_length_ :]

    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == f.window_length_
