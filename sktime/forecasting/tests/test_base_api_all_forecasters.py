#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"
__all__ = [
    "test_not_fitted_error",
    "test_predict_time_index",
    "test_update_predict_check_predicted_indices",
]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_selection import RollingWindowSplit
from sktime.utils import all_estimators
from sktime.utils.testing import _construct_instance
from sktime.utils.validation.forecasting import validate_fh

# get all forecasters
Forecasters = [e[1] for e in all_estimators(type_filter="forecaster")]

# testing grid
WINDOW_LENGTHS = [1, 3, 5]
STEP_LENGTHS = [1, 3, 5]
FHS = [1, 3, np.arange(1, 5)]
fh = FHS[0]

# testing data
n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


########################################################################################################################
# test clone
@pytest.mark.parametrize("Forecaster", Forecasters)
def test_clone(Forecaster):
    f = _construct_instance(Forecaster)
    clone(f)


########################################################################################################################
# fit, set_params and update return self
@pytest.mark.parametrize("Forecaster", Forecasters)
def test_fit_set_params_and_update_return_self(Forecaster):
    f = _construct_instance(Forecaster)
    ret = f.fit(y_train, fh)
    assert ret == f

    ret = f.update(y_test)
    assert ret == f

    ret = f.set_params()
    assert ret == f


########################################################################################################################
# not fitted error
@pytest.mark.parametrize("Forecaster", Forecasters)
def test_not_fitted_error(Forecaster):
    f = _construct_instance(Forecaster)
    with pytest.raises(NotFittedError):
        f.predict(fh=1)

    with pytest.raises(NotFittedError):
        f.update(y_test)

    with pytest.raises(NotFittedError):
        cv = RollingWindowSplit(fh=1, window_length=1)
        f.update_predict(y_test, cv=cv)


########################################################################################################################
# predict
# predicted time index
@pytest.mark.parametrize("Forecaster", Forecasters)
@pytest.mark.parametrize("fh", FHS)
def test_predict_time_index(Forecaster, fh):
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    y_pred = f.predict()

    fh = validate_fh(fh)
    np.testing.assert_array_equal(y_pred.index.values, y_train.iloc[-1] + fh)


########################################################################################################################
# update_predict
def compute_expected_index_from_update_predict(y_test, fh, step_length):
    """Helper function to compute expected time index from `update_predict`"""
    # time points at which to make predictions
    predict_at_all = np.arange(y_test.index.values[0] - 1, y_test.index.values[-1], step_length)

    # only predict at time points if all steps in fh can be predicted within y_test
    predict_at = predict_at_all[np.isin(predict_at_all + max(fh), y_test)]
    n_predict_at = len(predict_at)

    # all time points predicted, including duplicates from overlapping fhs
    broadcast_fh = np.repeat(fh, n_predict_at).reshape(len(fh), n_predict_at)
    points_predicted = predict_at + broadcast_fh

    # return only unique time points
    return np.unique(points_predicted)


# check if predicted time index is correct
@pytest.mark.parametrize("Forecaster", Forecasters)
@pytest.mark.parametrize("fh", FHS)
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", STEP_LENGTHS)
def test_update_predict_check_predicted_indices(Forecaster, fh, window_length, step_length):
    # initiate cv with different fh, so that out window in temporal cv does not contain fh
    cv = RollingWindowSplit(fh, window_length=window_length, step_length=step_length)
    f = _construct_instance(Forecaster)
    f.fit(y_train, fh)
    y_pred = f.update_predict(y_test, cv=cv)

    pred_index = y_pred.index.values
    expected_index = compute_expected_index_from_update_predict(y_test, f.fh, step_length)
    np.testing.assert_array_equal(pred_index, expected_index)
