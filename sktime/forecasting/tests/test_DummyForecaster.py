#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.dummy import DummyForecaster
from sktime.forecasting.model_selection import RollingWindowSplit

n_timepoints = 30
n_train = 20
s = pd.Series(np.arange(n_timepoints))
y_train = s.iloc[:n_train]
y_test = s.iloc[n_train:]


# update_predict
# inconsistent fhs error
@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("window_length", [1, 3, 5])
@pytest.mark.parametrize("step_length", [1, 3, 5])
def test_update_predict_inconsistent_fhs(fh, window_length, step_length):
    # check user warning if fh passed through cv is different from fh seen in fit
    cv = RollingWindowSplit(fh + 1, window_length=window_length, step_length=step_length)
    f = DummyForecaster(strategy="last")
    f.fit(y_train, fh)
    with pytest.warns(UserWarning):
        f.update_predict(y_test, cv=cv)


# updating and forecasts
def compute_expected_index_from_update_predict(y_test, fh, step_length):
    """Helper function to compute expected time index from `update_predict`"""
    # points at which to make predictions
    predict_at_all = np.arange(y_test.index.values[0] - 1, y_test.index.values[-1], step_length)

    # only predict if all steps in fh can be predicted
    predict_at = predict_at_all[np.isin(predict_at_all + max(fh), y_test)]
    n_predict_at = len(predict_at)

    # points predicted
    broadcast_fh = np.repeat(fh, n_predict_at).reshape(len(fh), n_predict_at)
    points_predicted = predict_at + broadcast_fh

    # return only unique points
    return np.unique(points_predicted)


@pytest.mark.parametrize("fh", [1, 3, np.arange(1, 5)])
@pytest.mark.parametrize("window_length", [1, 3, 5])
@pytest.mark.parametrize("step_length", [1, 3, 5])
def test_update_predict_indices(fh, window_length, step_length):
    # initiate cv with different fh, so that out window in temporal cv does not contain fh
    cv = RollingWindowSplit(fh, window_length=window_length, step_length=step_length)
    f = DummyForecaster(strategy="last")
    f.fit(y_train)
    y_pred = f.update_predict(y_test, cv=cv)

    # check time index
    pred_index = y_pred.index.values
    expected_index = compute_expected_index_from_update_predict(y_test, f.fh, step_length)
    np.testing.assert_array_equal(pred_index, expected_index)
