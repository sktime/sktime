# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for native time-series regression grid search."""

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sktime.regression.dummy import DummyRegressor
from sktime.regression.model_selection import TSRGridSearchCV
from sktime.utils._testing.panel import make_regression_problem


def test_grid_search_uses_native_evaluate():
    """Grid search should expose a fitted sktime estimator, not sklearn GridSearchCV."""
    X, y = make_regression_problem(
        n_instances=8, n_columns=1, n_timepoints=3, random_state=0
    )
    search = TSRGridSearchCV(
        DummyRegressor(),
        {"strategy": ["mean", "median"]},
        scoring=r2_score,
        cv=KFold(n_splits=2, shuffle=False),
    )

    search.fit(X, y)

    assert isinstance(search.estimator_, DummyRegressor)
    assert search.best_params_ in search.cv_results_["params"]
    assert (
        search.best_score_ == search.cv_results_["mean_test_score"][search.best_index_]
    )
    assert "mean_test_r2_score" in search.cv_results_
