"""Tests for TSRGridSearchCV."""

__author__ = ["yash-sangwan"]

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from sktime.datasets import load_unit_test
from sktime.regression.dummy import DummyRegressor
from sktime.regression.model_evaluation import evaluate
from sktime.regression.model_selection import TSRGridSearchCV
from sktime.tests.test_switch import run_test_for_class

PARAM_GRID = {"strategy": ["constant"], "constant": [1.0, 2.0, 100.0]}
CV = KFold(n_splits=2, shuffle=False)

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TSRGridSearchCV),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _fit_tuner(**kwargs):
    X, y = load_unit_test(split="train")
    y = y.astype("float")
    kwargs.setdefault("cv", CV)
    return TSRGridSearchCV(DummyRegressor(), PARAM_GRID, **kwargs).fit(X, y), X, y


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
    assert tuner.scorer_ is r2_score
    assert not tuner.multimetric_
    assert tuner.refit_time_ > 0


def test_delegate_is_the_fitted_best_estimator():
    """The delegate is best_estimator_, fitted, and carries the best parameters."""
    tuner, X, y = _fit_tuner()

    assert tuner._get_delegate() is tuner.best_estimator_
    assert isinstance(tuner.best_estimator_, DummyRegressor)
    assert tuner.best_estimator_.is_fitted


def test_predict_delegates():
    """predict returns the output of best_estimator_."""
    tuner, X, y = _fit_tuner()

    y_pred = tuner.predict(X)
    assert len(y_pred) == len(y)
    np.testing.assert_allclose(
        np.ravel(y_pred), np.ravel(tuner.best_estimator_.predict(X))
    )


def test_scores_match_evaluate():
    """mean_test_score of each candidate equals a manual evaluate run."""
    tuner, X, y = _fit_tuner()
    cv_results = tuner.cv_results_

    for i, params in enumerate(cv_results["params"]):
        expected = evaluate(
            DummyRegressor().set_params(**params), cv=CV, X=X, y=y, scoring=r2_score
        )
        assert np.isclose(
            cv_results["mean_test_score"][i], expected["test_r2_score"].mean()
        )


def test_default_scoring_is_r2():
    """The default metric stays r2_score, higher-is-better."""
    tuner, _, _ = _fit_tuner()
    scores = tuner.cv_results_["mean_test_score"]

    assert tuner.scorer_ is r2_score
    assert scores[tuner.best_index_] == max(scores)


def test_loss_metric_ranks_lower_as_better():
    """A loss metric selects the candidate with the lowest mean score."""
    tuner, _, _ = _fit_tuner(scoring=mean_squared_error)
    scores = tuner.cv_results_["mean_test_score"]

    assert scores[tuner.best_index_] == min(scores)
    assert scores[tuner.best_index_] < max(scores)


def test_negative_scorer_name_keeps_sklearn_sign():
    """neg_ scorer names report negative values, and select the same candidate."""
    by_string, _, _ = _fit_tuner(scoring="neg_mean_squared_error")
    by_callable, _, _ = _fit_tuner(scoring=mean_squared_error)

    assert all(score <= 0 for score in by_string.cv_results_["mean_test_score"])
    assert by_string.best_params_ == by_callable.best_params_


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


def test_get_fitted_params_contains_best_params():
    """get_fitted_params exposes the best parameters, and the delegate's."""
    tuner, _, _ = _fit_tuner()
    fitted_params = tuner.get_fitted_params()

    assert fitted_params["constant"] == tuner.best_params_["constant"]
    assert "best_estimator" in fitted_params


def test_get_fitted_params_without_refit():
    """With refit=False there are no fitted params, but no exception either."""
    tuner, _, _ = _fit_tuner(refit=False)
    fitted_params = tuner.get_fitted_params()

    assert isinstance(fitted_params, dict)
    assert fitted_params["constant"] == tuner.best_params_["constant"]


def test_no_sklearn_gridsearch_delegate():
    """The tuner does not wrap a sklearn GridSearchCV."""
    from sklearn.model_selection import GridSearchCV

    tuner, _, _ = _fit_tuner()

    assert not isinstance(tuner.best_estimator_, GridSearchCV)
    assert not hasattr(tuner, "estimator_")
