# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test regression tuners."""

__author__ = ["atikulmunna"]

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

from sktime.datasets import load_unit_test
from sktime.regression.dummy import DummyRegressor
from sktime.regression.model_evaluation import evaluate
from sktime.regression.model_selection import TSRGridSearchCV
from sktime.tests.test_switch import run_test_for_class

PARAM_GRID = {"strategy": ["mean", "median"]}
CV = KFold(n_splits=2, shuffle=False)


def _load_regression_data():
    """Load small panel data with float targets for regression tests."""
    X, y = load_unit_test(split="train")
    return X, y.astype("float")


@pytest.mark.skipif(
    not run_test_for_class(TSRGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_fit_and_attributes():
    """Test that fitted attributes are set and consistent."""
    X, y = _load_regression_data()
    tuner = TSRGridSearchCV(
        DummyRegressor(),
        param_grid=PARAM_GRID,
        cv=CV,
    )
    tuner.fit(X, y)

    n_candidates = len(ParameterGrid(PARAM_GRID))
    assert len(tuner.cv_results_) == n_candidates

    expected_columns = {
        "mean_test_mean_squared_error",
        "mean_fit_time",
        "mean_pred_time",
        "params",
        "rank_test_mean_squared_error",
    }
    assert expected_columns.issubset(tuner.cv_results_.columns)

    assert tuner.best_params_ in list(ParameterGrid(PARAM_GRID))
    assert tuner.n_splits_ == 2
    assert tuner.scorer_ is mean_squared_error

    y_pred = tuner.predict(X)
    assert len(y_pred) == len(y)


@pytest.mark.skipif(
    not run_test_for_class(TSRGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_ranks_lower_mse_best():
    """Test that ranking treats mean_squared_error as lower-is-better."""
    X, y = _load_regression_data()
    tuner = TSRGridSearchCV(
        DummyRegressor(),
        param_grid=PARAM_GRID,
        cv=CV,
    )
    tuner.fit(X, y)

    results = tuner.cv_results_
    best_row = results.loc[results["rank_test_mean_squared_error"] == 1].iloc[0]
    assert best_row["mean_test_mean_squared_error"] == (
        results["mean_test_mean_squared_error"].min()
    )
    assert tuner.best_score_ == results["mean_test_mean_squared_error"].min()


@pytest.mark.skipif(
    not run_test_for_class(TSRGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_scores_match_evaluate():
    """Test that cv_results_ scores equal manual evaluate scores per candidate."""
    X, y = _load_regression_data()
    tuner = TSRGridSearchCV(
        DummyRegressor(),
        param_grid=PARAM_GRID,
        cv=CV,
    )
    tuner.fit(X, y)

    for _, row in tuner.cv_results_.iterrows():
        estimator = DummyRegressor().set_params(**row["params"])
        expected = evaluate(estimator, cv=CV, X=X, y=y, scoring=mean_squared_error)
        expected_mean = expected["test_mean_squared_error"].mean()
        assert np.isclose(row["mean_test_mean_squared_error"], expected_mean)


@pytest.mark.skipif(
    not run_test_for_class(TSRGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_refit_false():
    """Test that refit=False tunes but refuses to predict."""
    X, y = _load_regression_data()
    tuner = TSRGridSearchCV(
        DummyRegressor(),
        param_grid=PARAM_GRID,
        cv=CV,
        refit=False,
    )
    tuner.fit(X, y)

    assert tuner.best_params_ in list(ParameterGrid(PARAM_GRID))

    with pytest.raises(RuntimeError, match="refit must be True"):
        tuner.predict(X)
