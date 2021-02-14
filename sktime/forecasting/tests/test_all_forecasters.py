#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# test API provided through BaseForecaster

__author__ = ["Markus Löning"]
__all__ = [
    "test_raises_not_fitted_error",
    "test_score",
    "test_predict_time_index",
    "test_update_predict_predicted_index",
    "test_update_predict_predicted_index_update_params",
    "test_y_multivariate_raises_error",
    "test_get_fitted_params",
    "test_predict_time_index_in_sample_full",
    "test_predict_pred_interval",
    "test_update_predict_single",
    "test_y_invalid_type_raises_error",
    "test_predict_time_index_with_X",
    "test_X_invalid_type_raises_error",
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
from sktime.forecasting.tests._config import VALID_INDEX_FH_COMBINATIONS
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils import all_estimators
from sktime.utils._testing.estimator_checks import _construct_instance
from sktime.utils._testing.forecasting import _assert_correct_pred_time_index
from sktime.utils._testing.forecasting import _get_expected_index_for_update_predict
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.series import _make_series
from sktime.utils.validation.forecasting import check_fh

# get all forecasters
FORECASTERS = all_estimators(estimator_types="forecaster", return_names=False)
FH0 = 1
INVALID_INPUT_TYPES = [np.empty(20), list(), tuple()]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
def test_get_fitted_params(Forecaster):
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
    f = _construct_instance(Forecaster)
    with pytest.raises(ValueError, match=r"univariate"):
        f.fit(y, fh=FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("y", INVALID_INPUT_TYPES)
def test_y_invalid_type_raises_error(Forecaster, y):
    f = _construct_instance(Forecaster)
    with pytest.raises(TypeError, match=r"type"):
        f.fit(y, fh=FH0)


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("X", INVALID_INPUT_TYPES)
def test_X_invalid_type_raises_error(Forecaster, X):
    f = _construct_instance(Forecaster)
    try:
        with pytest.raises(TypeError, match=r"type"):
            f.fit(y_train, X, fh=FH0)
    except NotImplementedError as e:
        msg = str(e).lower()
        assert "exogenous" in msg


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

    # Some estimators may not support all time index types and fh types, hence we
    # need to catch NotImplementedErrors.
    try:
        f.fit(y_train, fh=fh)
        y_pred = f.predict()
        _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("steps", TEST_OOS_FHS)  # fh steps
def test_predict_time_index_with_X(Forecaster, index_type, fh_type, is_relative, steps):
    # Check that predicted time index matches forecasting horizon.
    y, X = make_forecasting_problem(index_type=index_type, make_X=True)
    cutoff = y.index[len(y) // 2]
    fh = _make_fh(cutoff, steps, fh_type, is_relative)

    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)

    f = _construct_instance(Forecaster)
    # Some estimators may not support all time index types and fh types, hence we
    # need to catch NotImplementedErrors.
    try:
        f.fit(y_train, X_train, fh=fh)
        y_pred = f.predict(X=X_test)
        _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
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
    # Some estimators may not support all time index types and fh types, hence we
    # need to catch NotImplementedErrors.
    try:
        f.fit(y_train, fh=fh)
        y_pred = f.predict()
        _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
    except NotImplementedError:
        pass


def _check_pred_ints(pred_ints: list, y_train: pd.Series, y_pred: pd.Series, fh):
    # make iterable
    if isinstance(pred_ints, pd.DataFrame):
        pred_ints = [pred_ints]

    for pred_int in pred_ints:
        # check column naming convention
        assert list(pred_int.columns) == ["lower", "upper"]

        # check time index
        _assert_correct_pred_time_index(pred_int.index, y_train.index[-1], fh)

        # check values
        assert np.all(pred_int["upper"] > y_pred)
        assert np.all(pred_int["lower"] < y_pred)

        # check if errors are weakly monotonically increasing
        # pred_errors = y_pred - pred_int["lower"]
        # # assert pred_errors.is_mononotic_increasing
        # assert np.all(
        #     pred_errors.values[1:].round(4) >= pred_errors.values[:-1].round(4)
        # )


@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("alpha", TEST_ALPHAS)
def test_predict_pred_interval(Forecaster, fh, alpha):
    # Check prediction intervals.
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    try:
        y_pred, pred_ints = f.predict(return_pred_int=True, alpha=alpha)
        _check_pred_ints(pred_ints, y_train, y_pred, fh)

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
@pytest.mark.parametrize("update_params", [True, False])
def test_update_predict_single(Forecaster, fh, update_params):
    # Check correct time index of update-predict
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    y_pred = f.update_predict_single(y_test, update_params=update_params)
    _assert_correct_pred_time_index(y_pred.index, y_test.index[-1], fh)


def _check_update_predict_predicted_index(
    Forecaster, fh, window_length, step_length, update_params
):
    y = make_forecasting_problem(all_positive=True, index_type="datetime")
    y_train, y_test = temporal_train_test_split(y)
    cv = SlidingWindowSplitter(fh, window_length=window_length, step_length=step_length)
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh=fh)
    y_pred = f.update_predict(y_test, cv=cv, update_params=update_params)

    assert isinstance(y_pred, (pd.Series, pd.DataFrame))
    if isinstance(y_pred, pd.DataFrame):
        assert y_pred.shape[1] > 1
    expected = _get_expected_index_for_update_predict(y_test, fh, step_length)
    actual = y_pred.index
    np.testing.assert_array_equal(actual, expected)


# test with update_params=False and different values for steps_length
@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("update_params", [False])
def test_update_predict_predicted_index(
    Forecaster, fh, window_length, step_length, update_params
):
    _check_update_predict_predicted_index(
        Forecaster, fh, window_length, step_length, update_params
    )


# test with update_params=True and step_length=1
@pytest.mark.parametrize("Forecaster", FORECASTERS)
@pytest.mark.parametrize("fh", TEST_OOS_FHS)
@pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", [1])
@pytest.mark.parametrize("update_params", [True])
def test_update_predict_predicted_index_update_params(
    Forecaster, fh, window_length, step_length, update_params
):
    _check_update_predict_predicted_index(
        Forecaster, fh, window_length, step_length, update_params
    )
