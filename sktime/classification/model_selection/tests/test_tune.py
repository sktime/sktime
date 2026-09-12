# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for native time-series classification grid search."""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_selection import TSCGridSearchCV
from sktime.datasets import load_unit_test
from sktime.utils._testing.panel import make_classification_problem


def test_grid_search_uses_native_evaluate():
    """Grid search should expose a fitted sktime estimator, not sklearn GridSearchCV."""
    X, y = load_unit_test(split="train", return_X_y=True)
    search = TSCGridSearchCV(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior"]},
        scoring=accuracy_score,
        cv=StratifiedKFold(n_splits=2, shuffle=False),
    )

    search.fit(X, y)

    assert isinstance(search.estimator_, DummyClassifier)
    assert search.best_params_ in search.cv_results_["params"]
    assert (
        search.best_score_ == search.cv_results_["mean_test_score"][search.best_index_]
    )
    assert "mean_test_accuracy_score" in search.cv_results_


def test_grid_search_probability_scorer():
    """String scorers needing probabilities should use ``predict_proba``."""
    X, y = make_classification_problem(n_instances=12, random_state=0)
    search = TSCGridSearchCV(
        DummyClassifier(),
        {"strategy": ["prior"]},
        scoring="neg_brier_score",
        cv=StratifiedKFold(n_splits=2, shuffle=False),
    )

    search.fit(X, y)

    assert "mean_test_neg_brier_score" in search.cv_results_
