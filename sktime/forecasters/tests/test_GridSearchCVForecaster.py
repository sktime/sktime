#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from sktime.forecasters.compose import ReducedTimeSeriesRegressionForecaster
from sktime.forecasters.forecasters import DummyForecaster
from sktime.forecasters.model_selection import ForecastingGridSearchCV, RollingWindowSplit
from sktime.performance_metrics.forecasting import smape_score
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser
from sktime.datasets import load_shampoo_sales


@pytest.mark.parametrize("fh", [
    np.array([1], dtype=np.int),
    np.arange(2, 4)
])
def test_ForecastingGridSearchCV_best_params(fh):
    y = load_shampoo_sales()

    ts_regressor = Pipeline([
        ("tabularise", Tabulariser()),
        ("regress", LinearRegression())
    ])

    f = ReducedTimeSeriesRegressionForecaster(ts_regressor=ts_regressor)
    param_grid = {"window_length": [3, 5, 7]}
    cv = RollingWindowSplit(window_length=10, fh=fh)
    scoring = smape_score

    # using ForecastingGridSearchCV
    gscv = ForecastingGridSearchCV(f, param_grid, cv=cv, scoring=scoring, refit=False)
    gscv.fit(y, fh=fh)
    actual_scores = gscv.cv_results_["mean_test_score"]
    actual_best_param = gscv.best_params_["window_length"]

    # manual grid-search cv
    params = param_grid["window_length"]
    expected_scores = np.zeros(len(params))
    for i, param in enumerate(params):
        f.set_params(**{"window_length": param})

        scores = []
        for train, test in cv.split(y.index.values):
            y_train = y[train]
            y_test = y[test]
            f.fit(y_train, fh=fh)
            y_pred = f.predict(fh=fh)
            score = smape_score(y_test, y_pred)
            scores.append(score)
        expected_scores[i] = np.mean(scores)
    expected_best_param = params[expected_scores.argmax()]

    np.testing.assert_array_equal(actual_scores, expected_scores)
    assert actual_best_param == expected_best_param


@pytest.mark.parametrize("fh", [
    np.array([1], dtype=np.int),
    np.arange(2, 4)
])
def test_ForecastingGridSearchCV_predict(fh):
    y = load_shampoo_sales()

    f = DummyForecaster(strategy="last")
    param_grid = {"strategy": ["last", "mean"]}
    cv = RollingWindowSplit(window_length=10, fh=fh)
    scoring = smape_score
    gscv = ForecastingGridSearchCV(f, param_grid, cv=cv, scoring=scoring, refit=True)
    gscv.fit(y, fh=fh)

    y_pred = gscv.predict(fh=fh)
    assert y_pred.shape == (len(fh),)
