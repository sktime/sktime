#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import SingleWindowSplit, ForecastingGridSearchCV, SlidingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests import DEFAULT_FHS
from sktime.performance_metrics.forecasting import smape_loss


def compute_expected_gscv_scores(forecaster, cv, param_grid, y, score_func):
    training_window, validation_window = cv.split_initial(y)
    y_train, y_val = y.iloc[training_window], y.iloc[validation_window]

    scores = np.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        f = clone(forecaster)
        f.set_params(**params)
        f.fit(y_train)
        y_pred = f.update_predict(y_val, cv)
        y_test = y_val.loc[y_pred.index]
        scores[i] = score_func(y_test, y_pred)
    return scores


@pytest.mark.parametrize("fh", DEFAULT_FHS)
def test_gscv(fh):
    param_dict = {"window_length": [3, 5, 7]}
    param_grid = ParameterGrid(param_dict)

    y = load_airline()
    score_func = smape_loss
    f = NaiveForecaster(strategy="mean")
    cv = SingleWindowSplit(fh)
    gscv = ForecastingGridSearchCV(f, param_grid=param_dict, cv=cv, scoring=smape_loss)
    gscv.fit(y)

    # check scores
    gscv_scores = gscv.cv_results_["mean_test_score"]
    expected_scores = compute_expected_gscv_scores(f, cv, param_grid, y, score_func)
    np.testing.assert_array_equal(gscv_scores, expected_scores)

    # check best parameters
    assert gscv.best_params_ == param_grid[gscv_scores.argmin()]

    # check best forecaster is the one with best parameters
    assert {key: value for key, value in gscv.best_forecaster_.get_params().items() if
            key in gscv.best_params_.keys()} == gscv.best_params_
