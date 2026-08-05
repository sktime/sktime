# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test classification tuners."""

__author__ = ["atikulmunna"]

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, ParameterGrid

from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.classification.model_selection import TSCGridSearchCV
from sktime.classification.model_selection._tune import _metric_lower_is_better
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class

PARAM_GRID = {"strategy": ["most_frequent", "prior"]}
CV = KFold(n_splits=2, shuffle=False)


@pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_fit_and_attributes():
    """Test that fitted attributes are set and consistent."""
    X, y = load_unit_test(split="train")
    tuner = TSCGridSearchCV(
        DummyClassifier(),
        param_grid=PARAM_GRID,
        cv=CV,
    )
    tuner.fit(X, y)

    n_candidates = len(ParameterGrid(PARAM_GRID))
    assert len(tuner.cv_results_) == n_candidates

    expected_columns = {
        "mean_test_accuracy_score",
        "mean_fit_time",
        "mean_pred_time",
        "params",
        "rank_test_accuracy_score",
    }
    assert expected_columns.issubset(tuner.cv_results_.columns)

    assert tuner.best_params_ in list(ParameterGrid(PARAM_GRID))
    assert tuner.best_index_ in range(n_candidates)
    assert tuner.n_splits_ == 2
    assert tuner.scorer_ is accuracy_score
    assert not tuner.multimetric_
    assert hasattr(tuner, "refit_time_")

    best_row = tuner.cv_results_.iloc[tuner.best_index_]
    assert best_row["rank_test_accuracy_score"] == 1
    assert tuner.best_score_ == best_row["mean_test_accuracy_score"]

    y_pred = tuner.predict(X)
    assert len(y_pred) == len(y)

    y_proba = tuner.predict_proba(X)
    assert y_proba.shape[0] == len(y)


@pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_scores_match_evaluate():
    """Test that cv_results_ scores equal manual evaluate scores per candidate."""
    X, y = load_unit_test(split="train")
    tuner = TSCGridSearchCV(
        DummyClassifier(),
        param_grid=PARAM_GRID,
        cv=CV,
    )
    tuner.fit(X, y)

    for _, row in tuner.cv_results_.iterrows():
        estimator = DummyClassifier().set_params(**row["params"])
        expected = evaluate(estimator, cv=CV, X=X, y=y, scoring=accuracy_score)
        expected_mean = expected["test_accuracy_score"].mean()
        assert np.isclose(row["mean_test_accuracy_score"], expected_mean)


@pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_multiple_metrics():
    """Test that multiple metrics are computed and the first is used for ranking."""
    X, y = load_unit_test(split="train")
    tuner = TSCGridSearchCV(
        DummyClassifier(),
        param_grid=PARAM_GRID,
        scoring=[accuracy_score, balanced_accuracy_score],
        cv=CV,
    )
    tuner.fit(X, y)

    assert "mean_test_accuracy_score" in tuner.cv_results_.columns
    assert "mean_test_balanced_accuracy_score" in tuner.cv_results_.columns
    assert "rank_test_accuracy_score" in tuner.cv_results_.columns
    assert tuner.multimetric_
    assert tuner.scorer_ is accuracy_score


@pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_refit_false():
    """Test that refit=False tunes but refuses to predict."""
    X, y = load_unit_test(split="train")
    tuner = TSCGridSearchCV(
        DummyClassifier(),
        param_grid=PARAM_GRID,
        cv=CV,
        refit=False,
    )
    tuner.fit(X, y)

    assert tuner.best_params_ in list(ParameterGrid(PARAM_GRID))
    assert not hasattr(tuner, "refit_time_")

    with pytest.raises(RuntimeError, match="refit must be True"):
        tuner.predict(X)


@pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gridsearch_string_scoring_raises():
    """Test that string scoring raises an informative error."""
    X, y = load_unit_test(split="train")
    tuner = TSCGridSearchCV(
        DummyClassifier(),
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=CV,
    )
    with pytest.raises(TypeError, match="String scoring"):
        tuner.fit(X, y)


def test_metric_lower_is_better_heuristic():
    """Test the metric direction heuristic on common sklearn metrics."""
    from sklearn.metrics import (
        brier_score_loss,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    assert _metric_lower_is_better(mean_squared_error)
    assert _metric_lower_is_better(mean_absolute_error)
    assert _metric_lower_is_better(log_loss)
    assert _metric_lower_is_better(brier_score_loss)
    assert not _metric_lower_is_better(accuracy_score)
    assert not _metric_lower_is_better(r2_score)
