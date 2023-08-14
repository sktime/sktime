#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test grid search CV."""

__author__ = ["mloning", "fkiraly"]
__all__ = ["test_gscv", "test_rscv"]

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid, ParameterSampler

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    ForecastingSkoptSearchCV,
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
from sktime.performance_metrics.forecasting.probabilistic import CRPS, PinballLoss
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.impute import Imputer
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.validation._dependencies import _check_estimator_deps

TEST_METRICS = [MeanAbsolutePercentageError(symmetric=True), MeanSquaredError()]
TEST_METRICS_PROBA = [CRPS(), PinballLoss()]


def _get_expected_scores(forecaster, cv, param_grid, y, X, scoring):
    scores = np.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        f = forecaster.clone()
        f.set_params(**params)
        out = evaluate(f, cv, y, X=X, scoring=scoring)
        scores[i] = out.loc[:, f"test_{scoring.name}"].mean()
    return scores


def _check_fitted_params_keys(fitted_params):
    # ensure that the best_forecaster and best_score are in fitted_params
    assert "best_forecaster" in fitted_params.keys()
    assert "best_score" in fitted_params.keys()


def _check_cv(forecaster, tuner, cv, param_grid, y, X, scoring):
    actual = tuner.cv_results_[f"mean_test_{scoring.name}"]

    expected = _get_expected_scores(forecaster, cv, param_grid, y, X, scoring)
    np.testing.assert_array_equal(actual, expected)

    # Check if best parameters are selected.
    best_idx = tuner.best_index_
    assert best_idx == actual.argmin()

    fitted_params = tuner.get_fitted_params()
    assert param_grid[best_idx].items() <= fitted_params.items()


def _create_hierarchical_data():
    y = _make_hierarchical(
        random_state=TEST_RANDOM_SEEDS[0],
        hierarchy_levels=(2, 2),
        min_timepoints=15,
        max_timepoints=15,
    )
    X = _make_hierarchical(
        random_state=TEST_RANDOM_SEEDS[1],
        hierarchy_levels=(2, 2),
        min_timepoints=15,
        max_timepoints=15,
    )
    return y, X


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
    SlidingWindowSplitter(fh=1, initial_window=12, step_length=3),
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
    _check_fitted_params_keys(gscv.get_fitted_params())


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

    Tests that ForecastingRandomizedSearchCV successfully searches the parameter
    distributions to identify the best parameter set
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
    y, X = _create_hierarchical_data()
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


@pytest.mark.skipif(
    not _check_estimator_deps(ARIMA, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("scoring", TEST_METRICS_PROBA)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
def test_gscv_proba(cv, scoring, error_score):
    """Test ForecastingGridSearchCV with probabilistic metrics."""
    y = load_airline()[:36]

    forecaster = ARIMA()
    param_grid = {"order": [(1, 0, 0), (1, 1, 0)]}

    gscv = ForecastingGridSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    gscv.fit(y)

    param_grid = ParameterGrid(param_grid)
    _check_cv(forecaster, gscv, cv, param_grid, y, None, scoring)
    _check_fitted_params_keys(gscv.get_fitted_params())


@pytest.mark.skipif(
    not _check_estimator_deps(ForecastingSkoptSearchCV, severity="none"),
    reason="skip test if required soft dependency not compatible",
)
@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("n_iter", TEST_N_ITERS)
def test_skoptcv(forecaster, param_grid, cv, scoring, error_score, n_iter):
    """Test ForecastingSkoptSearchCV.

    Tests that ForecastingSkoptSearchCV successfully searches the
    parameter distributions to identify the best parameter set
    """
    # test for forecasting dataset
    y_forecasting, X_forecasting = load_longley()
    # test for hierarchical dataset
    y_hierarchical, X_hierarchical = _create_hierarchical_data()

    datasets = [(y_hierarchical, X_hierarchical), (y_forecasting, X_forecasting)]
    sscv = ForecastingSkoptSearchCV(
        forecaster,
        param_distributions=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
        n_iter=n_iter,
        random_state=42,
        n_jobs=-1,
    )
    for y, X in datasets:
        sscv.fit(y, X)
        param_distributions = list(sscv.cv_results_["params"])
        _check_cv(forecaster, sscv, cv, param_distributions, y, X, scoring)
        _check_fitted_params_keys(sscv.get_fitted_params())


@pytest.mark.skipif(
    not _check_estimator_deps(ForecastingSkoptSearchCV, severity="none"),
    reason="skip test if required soft dependency not compatible",
)
def test_skoptcv_multiple_forecaster():
    """Test ForecastingSkoptSearchCV with multiple forecasters and custom n_iter.

    Other behaviours are tested in test_skoptcv.
    """
    params_distributions = [
        {
            "forecaster": [NaiveForecaster(sp=12)],
            "forecaster__strategy": ["drift", "last", "mean"],
        },  # iterate twice
        (
            {
                "imputer__method": ["mean", "median"],
                "forecaster": [ExponentialSmoothing(sp=12)],
                "forecaster__trend": ["add", "mul"],
            },
            3,
        ),  # iterate thrice custom iteration
    ]
    cv = CVs[-1]
    y, X = load_longley()
    pipe = TransformedTargetForecaster(
        steps=[("imputer", Imputer()), ("forecaster", NaiveForecaster())]
    )
    sscv = ForecastingSkoptSearchCV(
        forecaster=pipe,
        param_distributions=params_distributions,
        cv=cv,
        n_jobs=-1,
        random_state=123,
        n_points=2,
        n_iter=2,
    )
    sscv.fit(y, X)
    assert len(sscv.cv_results_) == 5
