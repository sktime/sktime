#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["test_gscv_fit", "test_rscv_fit"]

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, ParameterSampler

from sktime.datasets import load_airline
from sktime.forecasting.compose import ReducedForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import TEST_OOS_FHS
from sktime.forecasting.tests._config import TEST_STEP_LENGTHS
from sktime.forecasting.tests._config import TEST_WINDOW_LENGTHS
from sktime.forecasting.tests._config import TEST_RANDOM_SEEDS
from sktime.forecasting.tests._config import TEST_N_ITERS
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import make_forecasting_scorer
from sktime.performance_metrics.forecasting import sMAPE
from sktime.transformations.series.detrend import Detrender


def compute_expected_gscv_scores(forecaster, cv, param_grid, y, scoring):
    training_window, test_window = cv.split_initial(y)
    y_train, y_test = y.iloc[training_window], y.iloc[test_window]

    scores = np.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        f = clone(forecaster)
        f.set_params(**params)
        f.fit(y_train, fh=cv.fh)
        y_pred = f.update_predict(y_test, cv)
        y_test_subset = y_test.loc[
            y_pred.index
        ]  # select only time points which we predicted
        scores[i] = scoring(y_test_subset, y_pred)
    return scores


@pytest.mark.parametrize(
    "forecaster, param_dict",
    [
        (NaiveForecaster(strategy="mean"), {"window_length": TEST_WINDOW_LENGTHS}),
        # atomic estimator
        (
            TransformedTargetForecaster(
                [  # composite estimator
                    ("t", Detrender(PolynomialTrendForecaster())),
                    ("f", ReducedForecaster(LinearRegression(), scitype="regressor")),
                ]
            ),
            {
                "f__window_length": TEST_WINDOW_LENGTHS,
                "f__step_length": TEST_STEP_LENGTHS,
            },
        ),  # multiple params
    ],
)
@pytest.mark.parametrize(
    "scoring",
    [sMAPE(), make_forecasting_scorer(mean_squared_error, greater_is_better=False)],
)
@pytest.mark.parametrize(
    "cv",
    [
        *[SingleWindowSplitter(fh=fh) for fh in TEST_OOS_FHS],
        # single split with multi-step fh
        SlidingWindowSplitter(fh=1, initial_window=50)
        # multiple splits with single-step fh
    ],
)
def test_gscv_fit(forecaster, param_dict, cv, scoring):
    param_grid = ParameterGrid(param_dict)

    y = load_airline()
    gscv = ForecastingGridSearchCV(
        forecaster, param_grid=param_dict, cv=cv, scoring=scoring
    )
    gscv.fit(y)

    # check scores
    gscv_scores = gscv.cv_results_[f"mean_test_{scoring.name}"]
    expected_scores = compute_expected_gscv_scores(
        forecaster, cv, param_grid, y, scoring
    )
    np.testing.assert_array_equal(gscv_scores, expected_scores)

    # check best parameters
    assert gscv.best_params_ == param_grid[gscv_scores.argmin()]

    # check best forecaster is the one with best parameters
    assert {
        key: value
        for key, value in gscv.best_forecaster_.get_params().items()
        if key in gscv.best_params_.keys()
    } == gscv.best_params_


@pytest.mark.parametrize(
    "forecaster, param_dict",
    [
        (NaiveForecaster(strategy="mean"), {"window_length": TEST_WINDOW_LENGTHS}),
        # atomic estimator
        (
            TransformedTargetForecaster(
                [  # composite estimator
                    ("t", Detrender(PolynomialTrendForecaster())),
                    ("f", ReducedForecaster(LinearRegression(), "regressor")),
                ]
            ),
            {
                "f__window_length": TEST_WINDOW_LENGTHS,
                "f__step_length": TEST_STEP_LENGTHS,
            },
        ),  # multiple params
    ],
)
@pytest.mark.parametrize(
    "scoring",
    [sMAPE(), make_forecasting_scorer(mean_squared_error, greater_is_better=False)],
)
@pytest.mark.parametrize(
    "cv",
    [
        *[SingleWindowSplitter(fh=fh) for fh in TEST_OOS_FHS],
        # single split with multi-step fh
        SlidingWindowSplitter(fh=1, initial_window=50)
        # multiple splits with single-step fh
    ],
)
@pytest.mark.parametrize(
    "n_iter",
    TEST_N_ITERS,
)
@pytest.mark.parametrize(
    "random_state",
    TEST_RANDOM_SEEDS,
)
def test_rscv_fit(forecaster, param_dict, cv, scoring, n_iter, random_state):
    """Tests that ForecastingRandomizedSearchCV successfully searches the
    parameter distributions to identify the best parameter set
    """
    # samples uniformly from param dict values
    param_distributions = ParameterSampler(
        param_dict, n_iter, random_state=random_state
    )

    y = load_airline()
    rscv = ForecastingRandomizedSearchCV(
        forecaster,
        param_distributions=param_dict,
        cv=cv,
        scoring=scoring,
        n_iter=n_iter,
        random_state=random_state,
    )
    rscv.fit(y)

    # check scores
    rscv_scores = rscv.cv_results_[f"mean_test_{scoring.name}"]
    # convert ParameterSampler to list to ensure consistent # of scores
    expected_scores = compute_expected_gscv_scores(
        forecaster, cv, list(param_distributions), y, scoring
    )
    np.testing.assert_array_equal(rscv_scores, expected_scores)

    # check best parameters
    assert rscv.best_params_ == list(param_distributions)[rscv_scores.argmin()]

    # check best forecaster is the one with best parameters
    assert {
        key: value
        for key, value in rscv.best_forecaster_.get_params().items()
        if key in rscv.best_params_.keys()
    } == rscv.best_params_
