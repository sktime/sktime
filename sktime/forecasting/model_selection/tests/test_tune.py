#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test grid search CV."""

__author__ = ["mloning"]
__all__ = ["test_gscv", "test_rscv"]

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid, ParameterSampler

from sktime.datasets import load_longley
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import (
    TEST_N_ITERS,
    TEST_OOS_FHS,
    TEST_RANDOM_SEEDS,
    TEST_WINDOW_LENGTHS_INT,
)
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from sktime.transformations.series.detrend import Detrender
from sktime.utils._testing.hierarchical import _make_hierarchical

TEST_METRICS = [MeanAbsolutePercentageError(symmetric=True), MeanSquaredError()]


def _get_expected_scores(forecaster, cv, param_grid, y, X, scoring):
    scores = np.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        f = forecaster.clone()
        f.set_params(**params)
        out = evaluate(f, cv, y, X=X, scoring=scoring)
        scores[i] = out.loc[:, f"test_{scoring.name}"].mean()
    return scores


def _check_cv(forecaster, tuner, cv, param_grid, y, X, scoring):
    actual = tuner.cv_results_[f"mean_test_{scoring.name}"]

    expected = _get_expected_scores(forecaster, cv, param_grid, y, X, scoring)
    np.testing.assert_array_equal(actual, expected)

    # Check if best parameters are selected.
    best_idx = tuner.best_index_
    assert best_idx == actual.argmin()

    fitted_params = tuner.get_fitted_params()
    assert param_grid[best_idx].items() <= fitted_params.items()


NAIVE = NaiveForecaster(strategy="mean")
NAIVE_GRID = {"window_length": TEST_WINDOW_LENGTHS_INT}
PIPE = TransformedTargetForecaster(
    [
        ("transformer", Detrender(PolynomialTrendForecaster())),
        ("forecaster", NaiveForecaster()),
    ]
)
PIPE_GRID = {
    "transformer__forecaster__degree": [1, 2],
    "forecaster__strategy": ["last", "mean"],
}
CVs = [
    *[SingleWindowSplitter(fh=fh) for fh in TEST_OOS_FHS],
    SlidingWindowSplitter(fh=1, initial_window=15),
]
ERROR_SCORES = [np.nan, "raise", 1000]


@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
def test_gscv(forecaster, param_grid, cv, scoring, error_score):
    """Test ForecastingGridSearchCV."""
    y, X = load_longley()
    gscv = ForecastingGridSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    gscv.fit(y, X)

    param_grid = ParameterGrid(param_grid)
    _check_cv(forecaster, gscv, cv, param_grid, y, X, scoring)

    fitted_params = gscv.get_fitted_params()
    assert "best_forecaster" in fitted_params.keys()
    assert "best_score" in fitted_params.keys()


@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("n_iter", TEST_N_ITERS)
@pytest.mark.parametrize("random_state", TEST_RANDOM_SEEDS)
def test_rscv(forecaster, param_grid, cv, scoring, error_score, n_iter, random_state):
    """Test ForecastingRandomizedSearchCV.

    Tests that ForecastingRandomizedSearchCV successfully searches the
    parameter distributions to identify the best parameter set
    """
    y, X = load_longley()
    rscv = ForecastingRandomizedSearchCV(
        forecaster,
        param_distributions=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
        n_iter=n_iter,
        random_state=random_state,
    )
    rscv.fit(y, X)

    param_distributions = list(
        ParameterSampler(param_grid, n_iter, random_state=random_state)
    )
    _check_cv(forecaster, rscv, cv, param_distributions, y, X, scoring)


@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
def test_gscv_hierarchical(forecaster, param_grid, cv, scoring, error_score):
    """Test ForecastingGridSearchCV."""
    y = _make_hierarchical(
        random_state=0, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )
    X = _make_hierarchical(
        random_state=42, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )

    gscv = ForecastingGridSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    gscv.fit(y, X)

    param_grid = ParameterGrid(param_grid)
    _check_cv(forecaster, gscv, cv, param_grid, y, X, scoring)
