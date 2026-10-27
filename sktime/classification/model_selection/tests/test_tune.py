"""Tests for TSCGridSearchCV."""

__author__ = ["yash-sangwan"]

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.classification.model_selection import TSCGridSearchCV
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class

PARAM_GRID = {"strategy": ["most_frequent", "prior", "stratified"]}
CV = KFold(n_splits=2, shuffle=False)

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TSCGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _fit_tuner(**kwargs):
    X, y = load_unit_test(split="train")
    kwargs.setdefault("cv", CV)
    return TSCGridSearchCV(DummyClassifier(), PARAM_GRID, **kwargs).fit(X, y), X, y


def test_fitted_attributes():
    """Fitted attributes are set, and consistent with each other."""
    tuner, X, y = _fit_tuner()
    cv_results = tuner.cv_results_

    assert isinstance(cv_results, dict)
    assert len(cv_results["params"]) == 3
    assert tuner.best_params_ == cv_results["params"][tuner.best_index_]
    assert tuner.best_score_ == cv_results["mean_test_score"][tuner.best_index_]
    assert cv_results["rank_test_score"][tuner.best_index_] == 1
    assert tuner.n_splits_ == 2
    assert tuner.scorer_ is accuracy_score
    assert not tuner.multimetric_
    assert tuner.refit_time_ > 0


def test_delegate_is_the_fitted_best_estimator():
    """The delegate is best_estimator_, fitted, and carries the best parameters."""
    tuner, X, y = _fit_tuner()

    assert tuner._get_delegate() is tuner.best_estimator_
    assert isinstance(tuner.best_estimator_, DummyClassifier)
    assert tuner.best_estimator_.is_fitted
    assert (
        tuner.best_estimator_.get_params()["strategy"] == tuner.best_params_["strategy"]
    )


def test_predict_and_predict_proba_delegate():
    """predict and predict_proba return the output of best_estimator_."""
    tuner, X, y = _fit_tuner()

    y_pred = tuner.predict(X)
    np.testing.assert_array_equal(y_pred, tuner.best_estimator_.predict(X))

    y_proba = tuner.predict_proba(X)
    assert y_proba.shape == (len(y), len(np.unique(y)))
    np.testing.assert_allclose(y_proba, tuner.best_estimator_.predict_proba(X))


def test_scores_match_evaluate():
    """mean_test_score of each candidate equals a manual evaluate run."""
    tuner, X, y = _fit_tuner()
    cv_results = tuner.cv_results_

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


def test_string_scoring():
    """String scorer names are accepted, as before the native rewrite."""
    tuner, _, _ = _fit_tuner(scoring="accuracy")

    assert tuner.multimetric_ is False
    assert all(0 <= score <= 1 for score in tuner.cv_results_["mean_test_score"])


def test_multimetric_scoring():
    """Multiple metrics are reported, and the first ranks the candidates."""
    tuner, _, _ = _fit_tuner(scoring=[accuracy_score, log_loss])

    assert tuner.multimetric_
    assert set(tuner.scorer_) == {"accuracy_score", "log_loss"}
    assert "mean_test_accuracy_score" in tuner.cv_results_
    assert "mean_test_log_loss" in tuner.cv_results_
    assert tuner.cv_results_["rank_test_accuracy_score"][tuner.best_index_] == 1


def test_greater_is_better_override():
    """greater_is_better=False makes the tuner select the lowest score."""
    tuner, _, _ = _fit_tuner(greater_is_better=False)
    scores = tuner.cv_results_["mean_test_score"]

    assert scores[tuner.best_index_] == min(scores)


def test_refit_false_tunes_but_does_not_predict():
    """refit=False selects parameters, but refuses to predict."""
    tuner, X, _ = _fit_tuner(refit=False)

    assert tuner.best_params_ in tuner.cv_results_["params"]
    assert not tuner.best_estimator_.is_fitted
    assert not hasattr(tuner, "refit_time_")

    with pytest.raises(RuntimeError, match="refit must be True"):
        tuner.predict(X)
    with pytest.raises(RuntimeError, match="refit must be True"):
        tuner.predict_proba(X)


def test_refit_callable():
    """A callable refit selects the candidate, and no best score is reported."""
    tuner, _, _ = _fit_tuner(refit=lambda cv_results: 2)

    assert tuner.best_index_ == 2
    assert not hasattr(tuner, "best_score_")
    assert tuner.best_estimator_.is_fitted


def test_get_fitted_params_contains_best_params():
    """get_fitted_params exposes the best parameters, and the delegate's."""
    tuner, _, _ = _fit_tuner()
    fitted_params = tuner.get_fitted_params()

    assert fitted_params["strategy"] == tuner.best_params_["strategy"]
    assert "best_estimator" in fitted_params


def test_get_fitted_params_without_refit():
    """With refit=False there are no fitted params, but no exception either."""
    tuner, _, _ = _fit_tuner(refit=False)
    fitted_params = tuner.get_fitted_params()

    assert isinstance(fitted_params, dict)
    assert fitted_params["strategy"] == tuner.best_params_["strategy"]


def test_backend_gives_same_result():
    """Parallelization over candidates does not change the result."""
    sequential, _, _ = _fit_tuner()
    parallel, _, _ = _fit_tuner(backend="loky", backend_params={"n_jobs": 2})

    np.testing.assert_allclose(
        sequential.cv_results_["mean_test_score"],
        parallel.cv_results_["mean_test_score"],
    )
    assert parallel.best_params_ == sequential.best_params_


# todo 1.3.0: remove this test together with the n_jobs and pre_dispatch parameters
@pytest.mark.parametrize("param", ["n_jobs", "pre_dispatch"])
def test_deprecated_parallel_params(param):
    """n_jobs and pre_dispatch warn, and still parallelize as before."""
    via_backend, _, _ = _fit_tuner(backend="loky", backend_params={param: 2})

    with pytest.warns(DeprecationWarning, match="sktime 1.3.0"):
        deprecated, _, _ = _fit_tuner(**{param: 2})

    np.testing.assert_allclose(
        via_backend.cv_results_["mean_test_score"],
        deprecated.cv_results_["mean_test_score"],
    )
    assert deprecated.best_params_ == via_backend.best_params_


# todo 1.3.0: remove this test together with the return_train_score parameter
def test_return_train_score_is_deprecated():
    """return_train_score warns when True, and no train scores are computed."""
    default, _, _ = _fit_tuner()

    with pytest.warns(DeprecationWarning, match="sktime 1.3.0"):
        deprecated, _, _ = _fit_tuner(return_train_score=True)

    np.testing.assert_allclose(
        default.cv_results_["mean_test_score"],
        deprecated.cv_results_["mean_test_score"],
    )
    assert not any("train" in key for key in deprecated.cv_results_)


def test_no_sklearn_gridsearch_delegate():
    """The tuner does not wrap a sklearn GridSearchCV."""
    from sklearn.model_selection import GridSearchCV

    tuner, _, _ = _fit_tuner()

    assert not isinstance(tuner.best_estimator_, GridSearchCV)
    assert not hasattr(tuner, "estimator_")
