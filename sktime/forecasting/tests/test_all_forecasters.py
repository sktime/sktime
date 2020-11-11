#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# test API provided through BaseForecaster

__author__ = ["Markus Löning"]
__all__ = [
    "test_raises_not_fitted_error",
    "test_score",
    "test_predict_time_index",
    "test_update_predict_predicted_indices",
    "test_y_multivariate_raises_error",
    "test_fitted_params",
    "test_predict_time_index_in_sample_full",
    "test_predict_pred_interval",
    "test_update_predict_single",
]

import numpy as np
import pandas as pd
import pytest

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import TEST_ALPHAS
from sktime.forecasting.tests._config import TEST_FHS
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_STEP_LENGTHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.forecasting.tests._config import TEST_YS
from sktime.forecasting.tests._config import VALID_INDEX_FH_COMBINATIONS
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils import all_estimators
from sktime.utils._testing import _construct_instance
from sktime.utils._testing import _make_series
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.forecasting import assert_correct_pred_time_index
from sktime.utils._testing.forecasting import get_expected_index_for_update_predict
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.validation.forecasting import check_fh

# get all forecasters
FORECASTERS = all_estimators(estimator_types="forecaster", return_names=False)
FH0 = 1

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
def test_fitted_params(Forecaster):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=FH0)
    try:
        params = f.get_fitted_params()
        assert isinstance(params, dict)

    except NotImplementedError:
        pass


@pytest.mark.parametrize("Forecaster", FORECASTERS)
def test_raises_not_fitted_error(Forecaster):
    # We here check extra method of the forecaster API: update and update_predict.
    f = _construct_instance(Forecaster)

    # predict is check in test suite for all estimators
    with pytest.raises(NotFittedError):
        f.update(y_test, update_params=False)

    with pytest.raises(NotFittedError):
        cv = SlidingWindowSplitter(fh=1, window_length=1)
        f.update_predict(y_test, cv=cv)

    try:
        with pytest.raises(NotFittedError):
            f.get_fitted_params()
    except NotImplementedError:
        pass


@pytest.mark.parametrize("Forecaster", FORECASTERS)
def test_y_multivariate_raises_error(Forecaster):
    # Check that multivariate y raises an appropriate error message.
    y = _make_series(n_columns=2)
    with pytest.raises(ValueError, match=r"univariate"):
        f = _construct_instance(Forecaster)
        f.fit(y, fh=FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("y", [np.empty(20), list(), tuple()])
def test_y_invalid_type_raises_error(Forecaster, y):
    with pytest.raises(TypeError, match=r"type"):
        f = _construct_instance(Forecaster)
        f.fit(y, fh=FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("steps", TEST_FHS)  # fh steps
def test_predict_time_index(Forecaster, index_type, fh_type, is_relative, steps):
    # Check that predicted time index matches forecasting horizon.
    y_train = make_forecasting_problem(index_type=index_type)
    cutoff = y_train.index[-1]
    fh = _make_fh(cutoff, steps, fh_type, is_relative)
    f = _construct_instance(Forecaster)
    try:
        f.fit(y_train, fh=fh)
        y_pred = f.predict()
        assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
def test_predict_time_index_in_sample_full(
    Forecaster, index_type, fh_type, is_relative
):
    # Check that predicted time index matched forecasting horizon for full in-sample
    # predictions.
    y_train = make_forecasting_problem(index_type=index_type)
    cutoff = y_train.index[-1]
    steps = -np.arange(len(y_train))  # full in-sample fh
    fh = _make_fh(cutoff, steps, fh_type, is_relative)
    f = _construct_instance(Forecaster)
    try:
        f.fit(y_train, fh=fh)
        y_pred = f.predict()
        assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
    except NotImplementedError:
        pass


def check_pred_ints(pred_ints, y_train, y_pred, fh):
    # make iterable
    if isinstance(pred_ints, pd.DataFrame):
        pred_ints = [pred_ints]

    for pred_int in pred_ints:
        assert list(pred_int.columns) == ["lower", "upper"]
        assert_correct_pred_time_index(pred_int.index, y_train.index[-1], fh)

        # check if errors are weakly monotonically increasing
        pred_errors = y_pred - pred_int["lower"]
        # assert pred_errors.is_mononotic_increasing
        assert np.all(
            pred_errors.values[1:].round(4) >= pred_errors.values[:-1].round(4)
        )


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("alpha", TEST_ALPHAS)
def test_predict_pred_interval(Forecaster, fh, alpha):
    # Check prediction intervals.
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    try:
        y_pred, pred_ints = f.predict(return_pred_int=True, alpha=alpha)
        check_pred_ints(pred_ints, y_train, y_pred, fh)

    except NotImplementedError:
        pass


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_score(Forecaster, fh):
    # Check score method
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    y_pred = f.predict()

    fh_idx = check_fh(fh).to_indexer()  # get zero based index
    expected = smape_loss(y_pred, y_test.iloc[fh_idx])

    # compare with actual score
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    actual = f.score(y_test.iloc[fh_idx], fh=fh)
    assert actual == expected


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
def test_update_predict_single(Forecaster, fh):
    # Check correct time index of update-predict
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    y_pred = f.update_predict_single(y_test)
    assert_correct_pred_time_index(y_pred.index, y_test.index[-1], fh)


def check_update_predict_y_pred(y_pred, y_test, fh, step_length):
    assert isinstance(y_pred, (pd.Series, pd.DataFrame))
    if isinstance(y_pred, pd.DataFrame):
        assert y_pred.shape[1] > 1
    expected_index = get_expected_index_for_update_predict(y_test, fh, step_length)
    np.testing.assert_array_equal(y_pred.index, expected_index)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("y", TEST_YS)
def test_update_predict_predicted_indices(
    Forecaster, fh, window_length, step_length, y
):
    y_train, y_test = temporal_train_test_split(y)
    cv = SlidingWindowSplitter(fh, window_length=window_length, step_length=step_length)
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    try:
        y_pred = f.update_predict(y_test, cv=cv)
        check_update_predict_y_pred(y_pred, y_test, fh, step_length)
    except NotImplementedError:
        pass
