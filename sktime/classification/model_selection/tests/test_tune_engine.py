"""Tests for the grid search engine shared by the classification and regression tuners.

The engine is tested directly here, the tuner classes are tested in ``test_tune``.
"""

__author__ = ["yash-sangwan"]

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold

from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.classification.model_selection._tune import (
    _resolve_cv,
    _run_grid_search,
)
from sktime.datasets import load_unit_test
from sktime.exceptions import NotFittedError
from sktime.tests.test_switch import run_test_module_changed

CLF_GRID = {"strategy": ["most_frequent", "prior", "stratified"]}
REG_GRID = {"strategy": ["constant"], "constant": [1.0, 2.0, 100.0]}
CV = KFold(n_splits=2, shuffle=False)

pytestmark = pytest.mark.skipif(
    not run_test_module_changed(["sktime.classification", "sktime.regression"]),
    reason="run test only if classification or regression code has changed",
)


def _failing_metric(y_true, y_pred):
    """Metric that always fails, to exercise the error_score paths."""
    raise ValueError("metric failure")


def _clf_data():
    return load_unit_test(split="train")


def _reg_data():
    X, y = load_unit_test(split="train")
    return X, y.astype("float")


def _clf_search(X, y, param_grid=None, estimator=None, **kwargs):
    kwargs.setdefault("cv", CV)
    return _run_grid_search(
        estimator=estimator if estimator is not None else DummyClassifier(),
        param_grid=param_grid if param_grid is not None else CLF_GRID,
        X=X,
        y=y,
        estimator_type="classifier",
        **kwargs,
    )


def _reg_search(X, y, **kwargs):
    # deferred import, sibling type modules must not cross-import at module level,
    # see sktime/tests/test_cross_module_imports.py
    from sktime.regression.dummy import DummyRegressor

    kwargs.setdefault("cv", CV)
    return _run_grid_search(
        estimator=DummyRegressor(),
        param_grid=REG_GRID,
        X=X,
        y=y,
        estimator_type="regressor",
        **kwargs,
    )


# cv resolution and split materialisation


def test_resolve_cv_int_is_stratified_for_classifiers():
    """int cv resolves to StratifiedKFold for classifiers, KFold for regressors."""
    y = np.array([0, 0, 0, 1, 1, 1])

    clf_splits = _resolve_cv(3, y, "classifier").splits
    reg_splits = _resolve_cv(3, y, "regressor").splits

    expected = list(StratifiedKFold(n_splits=3).split(np.arange(6), y))
    for actual, exp in zip(clf_splits, expected):
        np.testing.assert_array_equal(actual[1], exp[1])

    # every stratified test fold sees both classes, the KFold ones do not
    assert all(len(set(y[test])) == 2 for _, test in clf_splits)
    assert not all(len(set(y[test])) == 2 for _, test in reg_splits)


def test_resolve_cv_none_is_five_fold():
    """cv=None resolves to 5 folds, as in the sklearn grid search it replaces."""
    y = np.arange(10) % 2
    assert _resolve_cv(None, y, "classifier").get_n_splits() == 5


def test_resolve_cv_materialises_splits():
    """Splits are computed once, so a shuffling splitter yields stable folds."""
    y = np.arange(20) % 2
    cv = _resolve_cv(KFold(n_splits=3, shuffle=True), y, "classifier")

    first = [test for _, test in cv.split()]
    second = [test for _, test in cv.split()]

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_resolve_cv_accepts_iterable_of_splits():
    """An iterable of (train, test) splits is accepted."""
    y = np.arange(6) % 2
    splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    assert _resolve_cv(splits, y, "classifier").get_n_splits() == 1


# scores and cv_results_


def test_scores_match_evaluate():
    """mean_test_score of each candidate equals the mean of a manual evaluate run."""
    X, y = _clf_data()
    cv_results = _clf_search(X, y)["cv_results_"]

    for i, params in enumerate(cv_results["params"]):
        expected = evaluate(
            DummyClassifier().set_params(**params),
            cv=CV,
            X=X,
            y=y,
            scoring=accuracy_score,
        )
        assert np.isclose(
            cv_results["mean_test_score"][i], expected["test_accuracy_score"].mean()
        )


def test_cv_results_keys_single_metric():
    """cv_results_ has the sklearn keys and consistent shapes, for a single metric."""
    X, y = _clf_data()
    out = _clf_search(X, y)
    cv_results = out["cv_results_"]

    n_candidates = len(list(ParameterGrid(CLF_GRID)))
    expected = {
        "params",
        "param_strategy",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "split0_test_score",
        "split1_test_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    }
    assert expected.issubset(cv_results)
    assert all(len(cv_results[key]) == n_candidates for key in expected)
    assert out["n_splits_"] == 2
    assert not out["multimetric_"]

    split_scores = np.column_stack(
        [cv_results["split0_test_score"], cv_results["split1_test_score"]]
    )
    np.testing.assert_allclose(cv_results["mean_test_score"], split_scores.mean(axis=1))


def test_cv_results_param_columns_are_masked():
    """param_ columns are masked where a candidate does not set the parameter."""
    X, y = _clf_data()
    cv_results = _clf_search(
        X, y, param_grid=[{"strategy": ["prior"]}, {"random_state": [1]}]
    )["cv_results_"]

    assert list(cv_results["param_strategy"].mask) == [False, True]
    assert list(cv_results["param_random_state"].mask) == [True, False]


def test_multimetric_keys_and_selection():
    """Multi-metric scoring names the keys by metric, and ranks by the first."""
    X, y = _clf_data()
    out = _clf_search(X, y, scoring=[accuracy_score, log_loss])
    cv_results = out["cv_results_"]

    assert out["multimetric_"]
    assert "mean_test_score" not in cv_results
    for name in ["accuracy_score", "log_loss"]:
        assert f"mean_test_{name}" in cv_results
        assert f"rank_test_{name}" in cv_results

    best = out["best_index_"]
    assert cv_results["rank_test_accuracy_score"][best] == 1
    assert out["best_score_"] == cv_results["mean_test_accuracy_score"][best]


def test_multimetric_dict_scoring_uses_keys():
    """Dict scoring names the cv_results_ keys by the dict keys."""
    X, y = _clf_data()
    out = _clf_search(X, y, scoring={"acc": accuracy_score})
    assert "mean_test_acc" in out["cv_results_"]


# ranking direction


def test_rank_higher_is_better():
    """For a higher-is-better metric, rank 1 is the highest mean score."""
    X, y = _clf_data()
    out = _clf_search(X, y)
    cv_results = out["cv_results_"]

    best = out["best_index_"]
    assert cv_results["rank_test_score"][best] == 1
    assert cv_results["mean_test_score"][best] == max(cv_results["mean_test_score"])


def test_rank_lower_is_better():
    """For a lower-is-better metric, rank 1 is the lowest mean score."""
    X, y = _reg_data()
    out = _reg_search(X, y, scoring=mean_squared_error)
    cv_results = out["cv_results_"]

    best = out["best_index_"]
    assert cv_results["rank_test_score"][best] == 1
    assert cv_results["mean_test_score"][best] == min(cv_results["mean_test_score"])
    assert cv_results["mean_test_score"][best] < max(cv_results["mean_test_score"])


def test_string_scorer_keeps_sklearn_sign():
    """A neg_ scorer name reports negative values, and ranks higher-is-better."""
    X, y = _reg_data()
    out = _reg_search(X, y, scoring="neg_mean_squared_error")
    cv_results = out["cv_results_"]

    assert all(score <= 0 for score in cv_results["mean_test_score"])
    assert out["best_score_"] == max(cv_results["mean_test_score"])


def test_string_and_callable_select_same_candidate():
    """A metric and its negated scorer name select the same parameters."""
    X, y = _reg_data()
    by_callable = _reg_search(X, y, scoring=mean_squared_error)
    by_string = _reg_search(X, y, scoring="neg_mean_squared_error")

    assert by_callable["best_params_"] == by_string["best_params_"]


@pytest.mark.parametrize("greater_is_better", [True, False])
def test_greater_is_better_override(greater_is_better):
    """An explicit greater_is_better overrides the direction heuristic."""
    X, y = _reg_data()
    out = _reg_search(
        X, y, scoring=mean_squared_error, greater_is_better=greater_is_better
    )
    cv_results = out["cv_results_"]

    scores = cv_results["mean_test_score"]
    expected = max(scores) if greater_is_better else min(scores)
    assert cv_results["mean_test_score"][out["best_index_"]] == expected


def test_default_scoring_per_estimator_type():
    """Defaults are accuracy_score for classifiers and r2_score for regressors."""
    X, y = _clf_data()
    assert _clf_search(X, y)["scorer_"] is accuracy_score

    X, y = _reg_data()
    assert _reg_search(X, y)["scorer_"] is r2_score


# determinism, refit, and error paths


def test_search_is_deterministic():
    """Two runs on the same data give the same scores and best parameters."""
    X, y = _clf_data()
    first = _clf_search(X, y, estimator=DummyClassifier(random_state=42), cv=3)
    second = _clf_search(X, y, estimator=DummyClassifier(random_state=42), cv=3)

    np.testing.assert_allclose(
        first["cv_results_"]["mean_test_score"],
        second["cv_results_"]["mean_test_score"],
    )
    assert first["best_params_"] == second["best_params_"]


def test_candidates_share_the_same_folds():
    """All candidates are backtested on the same folds, also for a shuffling cv."""
    X, y = _clf_data()
    cv_results = _clf_search(
        X,
        y,
        estimator=DummyClassifier(strategy="most_frequent"),
        param_grid={"random_state": [1, 2, 3]},
        cv=KFold(n_splits=2, shuffle=True),
    )["cv_results_"]

    # the candidates predict identically, so identical folds imply identical scores
    for split in ["split0_test_score", "split1_test_score"]:
        assert len(set(cv_results[split])) == 1


def test_refit_string_selects_metric():
    """A string refit selects the named metric for multi-metric scoring."""
    X, y = _clf_data()
    out = _clf_search(X, y, scoring=[accuracy_score, log_loss], refit="log_loss")
    cv_results = out["cv_results_"]

    assert cv_results["rank_test_log_loss"][out["best_index_"]] == 1
    assert out["best_score_"] == cv_results["mean_test_log_loss"][out["best_index_"]]


def test_refit_callable_selects_index():
    """A callable refit selects the index, and no best score is reported."""
    X, y = _clf_data()
    out = _clf_search(X, y, refit=lambda results: 1)

    assert out["best_index_"] == 1
    assert "best_score_" not in out
    assert out["best_params_"] == out["cv_results_"]["params"][1]


def test_refit_string_unknown_metric_raises():
    """A string refit that names no metric raises an informative error."""
    X, y = _clf_data()
    with pytest.raises(ValueError, match="must be one of the metric names"):
        _clf_search(X, y, scoring=[accuracy_score, log_loss], refit="f1_score")


def test_empty_param_grid_raises():
    """An empty parameter grid raises an informative error."""
    X, y = _clf_data()
    with pytest.raises(ValueError, match="param_grid is empty"):
        _clf_search(X, y, param_grid=[])


def test_all_fits_failed_raises():
    """If every fit fails, an informative NotFittedError is raised."""
    X, y = _clf_data()
    with pytest.warns(UserWarning), pytest.raises(NotFittedError, match="all fits"):
        _clf_search(X, y, scoring=_failing_metric)


def test_error_score_is_used_for_failed_fits():
    """A numeric error_score is assigned to the folds that failed."""
    X, y = _clf_data()
    with pytest.warns(UserWarning):
        cv_results = _clf_search(X, y, scoring=_failing_metric, error_score=-1.0)[
            "cv_results_"
        ]

    assert all(score == -1.0 for score in cv_results["mean_test_score"])
