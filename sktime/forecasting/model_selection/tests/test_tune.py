#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test forecasting tuners."""

__author__ = ["mloning", "fkiraly"]


from functools import reduce
from typing import Union

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
    ForecastingOptunaSearchCV,
    ForecastingRandomizedSearchCV,
    ForecastingSkoptSearchCV,
)
from sktime.forecasting.model_selection._tune import BaseGridSearch
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
from sktime.split import SingleWindowSplitter, SlidingWindowSplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.impute import Imputer
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.parallel import _get_parallel_test_fixtures

TEST_METRICS = [MeanAbsolutePercentageError(symmetric=True), MeanSquaredError()]
TEST_METRICS_PROBA = [CRPS(), PinballLoss()]

TUNER_CLASSES = [
    BaseGridSearch,
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
    ForecastingSkoptSearchCV,
    ForecastingOptunaSearchCV,
]


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


def _create_hierarchical_data(n_columns=1):
    y = _make_hierarchical(
        random_state=TEST_RANDOM_SEEDS[0],
        hierarchy_levels=(2, 2),
        min_timepoints=15,
        max_timepoints=15,
        n_columns=n_columns,
    )
    X = _make_hierarchical(
        random_state=TEST_RANDOM_SEEDS[1],
        hierarchy_levels=(2, 2),
        min_timepoints=15,
        max_timepoints=15,
    )
    return y, X


# estimator fixtures used for tuning
# set_tags in NaiveForecaster ensures that it is univariate and broadcasts
# this is currently the case, but a future improved NaiveForecaster may reduce coverage
NAIVE = NaiveForecaster(strategy="mean").set_tags(**{"scitype:y": "univariate"})
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


@pytest.mark.skipif(
    not run_test_for_class(ForecastingGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("multivariate", [True, False])
def test_gscv(forecaster, param_grid, cv, scoring, error_score, multivariate):
    """Test ForecastingGridSearchCV."""
    if multivariate:
        X, y = load_longley()
    else:
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


@pytest.mark.skipif(
    not run_test_for_class(ForecastingRandomizedSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    _check_fitted_params_keys(rscv.get_fitted_params())


@pytest.mark.skipif(
    not run_test_for_class(ForecastingGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("n_cols", [1, 2])
def test_gscv_hierarchical(forecaster, param_grid, cv, scoring, error_score, n_cols):
    """Test ForecastingGridSearchCV."""
    y, X = _create_hierarchical_data(n_columns=n_cols)
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


@pytest.mark.skipif(
    not run_test_for_class([ForecastingGridSearchCV, ARIMA, CRPS, PinballLoss]),
    reason="run test only if softdeps are present and incrementally (if requested)",
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
    not run_test_for_class(ForecastingSkoptSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
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
    )
    for y, X in datasets:
        sscv.fit(y, X)
        param_distributions = list(sscv.cv_results_["params"])
        _check_cv(forecaster, sscv, cv, param_distributions, y, X, scoring)
        _check_fitted_params_keys(sscv.get_fitted_params())


@pytest.mark.skipif(
    not run_test_for_class(ForecastingSkoptSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
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
        random_state=123,
        n_points=2,
        n_iter=2,
    )
    sscv.fit(y, X)
    assert len(sscv.cv_results_) == 5


@pytest.mark.xfail
@pytest.mark.skipif(
    not run_test_for_class(ForecastingOptunaSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecaster, param_grid", [(NAIVE, NAIVE_GRID), (PIPE, PIPE_GRID)]
)
@pytest.mark.parametrize("scoring", TEST_METRICS)
@pytest.mark.parametrize("error_score", ERROR_SCORES)
@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("n_iter", TEST_N_ITERS)
@pytest.mark.parametrize("random_state", TEST_RANDOM_SEEDS)
def test_optuna(forecaster, param_grid, cv, scoring, error_score, n_iter, random_state):
    """Test TuneForecastingOptunaCV.

    Tests that TuneForecastingOptunaCV successfully searches the parameter
    distributions to identify the best parameter set
    """
    y, X = load_longley()
    rscv = ForecastingOptunaSearchCV(
        forecaster,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        error_score=error_score,
    )
    rscv.fit(y, X)

    param_distributions = list(
        ParameterSampler(param_grid, n_iter, random_state=random_state)
    )
    _check_cv(forecaster, rscv, cv, param_distributions, y, X, scoring)
    _check_fitted_params_keys(rscv.get_fitted_params())


BACKEND_TEST = _get_parallel_test_fixtures("estimator")


@pytest.mark.skipif(
    not run_test_for_class(ForecastingGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("backend_set", BACKEND_TEST)
def test_gscv_backends(backend_set):
    """Test ForecastingGridSearchCV."""
    backend = backend_set["backend"]
    backend_params = backend_set["backend_params"]

    y, X = load_longley()

    gscv = ForecastingGridSearchCV(
        PIPE,
        param_grid=PIPE_GRID,
        cv=CVs[0],
        scoring=TEST_METRICS[0],
        error_score=ERROR_SCORES[0],
        backend=backend,
        backend_params=backend_params,
    )
    gscv.fit(y, X)


TEST_PARAMS_DICT = PIPE_GRID

TEST_PARAMS_LIST = [
    {
        "window_length": [1, 2, 3],
        "strategy": ["last", "mean"],
        "transformer__degree": [1, 2, 3],
        "forecaster__strategy": ["last", "mean", "seasonal_last"],
    },
    {
        "window_length": [4, 5, 6],
        "forecaster__strategy": ["last", "mean"],
    },
]


@pytest.mark.skipif(
    not run_test_for_class([ForecastingGridSearchCV, ForecastingRandomizedSearchCV]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("return_n_best_forecasters", [-1, 0, 3])
@pytest.mark.parametrize(
    "Forecaster, kwargs",
    [
        (ForecastingGridSearchCV, {"param_grid": TEST_PARAMS_DICT}),
        (ForecastingGridSearchCV, {"param_grid": TEST_PARAMS_LIST}),
        (ForecastingRandomizedSearchCV, {"param_distributions": TEST_PARAMS_LIST}),
        (
            ForecastingRandomizedSearchCV,
            {"param_distributions": TEST_PARAMS_LIST, "n_iter": 100},
        ),
    ],
)
def test_return_n_best_forecasters(Forecaster, return_n_best_forecasters, kwargs):
    y, X = load_longley()
    searchCV = Forecaster(
        forecaster=PIPE,
        cv=CVs[0],
        **kwargs,
        return_n_best_forecasters=return_n_best_forecasters,
    )
    searchCV.fit(y, X)
    if return_n_best_forecasters == -1:

        def calculate_total_combinations(param_grid: Union[list[dict], dict]):
            if isinstance(param_grid, dict):
                return reduce(lambda x, y: x * y, [len(x) for x in param_grid.values()])
            elif isinstance(param_grid, list):
                return sum(calculate_total_combinations(i) for i in param_grid)
            else:
                error_message = "`param_grid` must be a dict or a list[dict]"
                raise ValueError(error_message)

        if "param_grid" in kwargs:
            total_combinations = calculate_total_combinations(kwargs["param_grid"])
            assert len(searchCV.n_best_forecasters_) == total_combinations
        else:
            try:
                assert len(searchCV.n_best_forecasters_) == searchCV.n_iter
            except AssertionError:
                total_combinations = calculate_total_combinations(
                    kwargs["param_distributions"]
                )
                assert len(searchCV.n_best_forecasters_) == total_combinations
    else:
        try:
            assert len(searchCV.n_best_forecasters_) == return_n_best_forecasters
        except AssertionError:
            key = (
                "param_distributions"
                if "param_distributions" in kwargs
                else "param_grid"
            )
            total_combinations = calculate_total_combinations(kwargs[key])
            assert len(searchCV.n_best_forecasters_) == total_combinations
