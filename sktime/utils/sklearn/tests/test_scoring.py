"""Tests for scoring resolution utilities in utils.sklearn."""

__author__ = ["yash-sangwan"]

import pickle

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    log_loss,
    make_scorer,
    max_error,
    mean_squared_error,
    r2_score,
)

from sktime.utils.sklearn._scoring import (
    _guess_greater_is_better,
    _resolve_scoring,
)

TABLE_HIGHER = [accuracy_score, r2_score, f1_score]
TABLE_LOWER = [mean_squared_error, log_loss, max_error]


@pytest.mark.parametrize("metric", TABLE_HIGHER)
def test_guess_direction_table_higher(metric):
    """Metrics in the lookup table with higher-is-better are recognized."""
    assert _guess_greater_is_better(metric)


@pytest.mark.parametrize("metric", TABLE_LOWER)
def test_guess_direction_table_lower(metric):
    """Metrics in the lookup table with lower-is-better are recognized."""
    assert not _guess_greater_is_better(metric)


def test_guess_direction_table_without_suffix():
    """Table entries without a known suffix are classified from the table.

    Without the table these would fall through to the lower-is-better default.
    """
    from sklearn.metrics import class_likelihood_ratios, matthews_corrcoef

    assert _guess_greater_is_better(matthews_corrcoef)
    assert not _guess_greater_is_better(class_likelihood_ratios)


@pytest.mark.parametrize(
    "suffix, expected",
    [("_score", True), ("_loss", False), ("_deviance", False), ("_error", False)],
)
def test_guess_direction_suffix(suffix, expected):
    """Metrics outside the lookup table are classified by their name suffix."""

    def metric(y_true, y_pred):
        return 0.0

    metric.__name__ = f"custom{suffix}"
    assert _guess_greater_is_better(metric) == expected


def test_guess_direction_unknown_name():
    """Metrics with an unknown name are assumed to be lower-is-better."""

    def metric(y_true, y_pred):
        return 0.0

    metric.__name__ = "my_metric"
    assert not _guess_greater_is_better(metric)


def test_guess_direction_no_name():
    """Callables without a name are assumed to be lower-is-better."""

    class NamelessMetric:
        def __call__(self, y_true, y_pred):
            return 0.0

    assert not _guess_greater_is_better(NamelessMetric())


@pytest.mark.parametrize("greater_is_better", [True, False])
def test_guess_direction_attribute(greater_is_better):
    """An explicit greater_is_better attribute takes precedence over the name."""

    def metric(y_true, y_pred):
        return 0.0

    metric.__name__ = "mean_squared_error"
    metric.greater_is_better = greater_is_better
    assert _guess_greater_is_better(metric) == greater_is_better


def test_resolve_default_classifier():
    """Default scoring for classifiers is accuracy_score, higher-is-better."""
    (resolved,) = _resolve_scoring(None, "classifier")
    assert resolved.metric is accuracy_score
    assert resolved.name == "accuracy_score"
    assert resolved.sign == 1
    assert resolved.greater_is_better


def test_resolve_default_regressor():
    """Default scoring for regressors is r2_score, higher-is-better."""
    (resolved,) = _resolve_scoring(None, "regressor")
    assert resolved.metric is r2_score
    assert resolved.name == "r2_score"
    assert resolved.sign == 1
    assert resolved.greater_is_better


def test_resolve_callable_is_passed_through():
    """Metric callables are passed through unchanged, with sign 1."""
    (resolved,) = _resolve_scoring(mean_squared_error, "regressor")
    assert resolved.metric is mean_squared_error
    assert resolved.name == "mean_squared_error"
    assert resolved.sign == 1
    assert not resolved.greater_is_better


def test_resolve_string_scorer_keeps_sklearn_sign():
    """String scorers keep the sklearn signed value, and rank higher-is-better."""
    (resolved,) = _resolve_scoring("neg_mean_squared_error", "regressor")
    assert resolved.name == "neg_mean_squared_error"
    assert resolved.sign == -1
    assert resolved.greater_is_better
    assert resolved.metric.__name__ == "mean_squared_error"


def test_resolve_string_scorer_positive_sign():
    """String scorers that are already higher-is-better have sign 1."""
    (resolved,) = _resolve_scoring("accuracy", "classifier")
    assert resolved.name == "accuracy"
    assert resolved.sign == 1
    assert resolved.greater_is_better


def test_resolve_string_scorer_binds_kwargs():
    """String scorers with kwargs bind them to the unwrapped metric."""
    (resolved,) = _resolve_scoring("f1_macro", "classifier")
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    assert resolved.metric(y_true, y_pred) == f1_score(y_true, y_pred, average="macro")


def test_resolve_scorer_object():
    """make_scorer objects are unwrapped, and their sign is respected."""
    (resolved,) = _resolve_scoring(
        make_scorer(mean_squared_error, greater_is_better=False), "regressor"
    )
    assert resolved.name == "mean_squared_error"
    assert resolved.sign == -1
    assert resolved.greater_is_better


def test_resolve_adapter_keeps_metric_signature():
    """The unwrapped metric presents the name and signature of the metric.

    ``evaluate`` identifies probabilistic metrics by name and signature, so the
    adapter must not hide either.
    """
    from inspect import signature

    (resolved,) = _resolve_scoring("neg_log_loss", "classifier")
    assert resolved.metric.__name__ == "log_loss"
    assert signature(resolved.metric) == signature(log_loss)


def test_resolve_adapter_is_picklable():
    """Unwrapped metrics survive pickling, as required by parallel backends."""
    (resolved,) = _resolve_scoring("neg_mean_squared_error", "regressor")
    metric = pickle.loads(pickle.dumps(resolved.metric))
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert metric(y_true, y_pred) == mean_squared_error(y_true, y_pred)


def test_resolve_list():
    """Lists resolve to one entry per metric, in order."""
    resolved = _resolve_scoring([accuracy_score, log_loss], "classifier")
    assert [entry.name for entry in resolved] == ["accuracy_score", "log_loss"]
    assert [entry.greater_is_better for entry in resolved] == [True, False]


def test_resolve_dict_uses_keys_as_names():
    """Dicts use their keys as metric names."""
    resolved = _resolve_scoring(
        {"acc": accuracy_score, "mse": mean_squared_error}, "classifier"
    )
    assert [entry.name for entry in resolved] == ["acc", "mse"]
    assert resolved[0].metric is accuracy_score


@pytest.mark.parametrize("greater_is_better", [True, False])
def test_resolve_greater_is_better_override(greater_is_better):
    """An explicit greater_is_better overrides the heuristic, for all metrics."""
    resolved = _resolve_scoring(
        [accuracy_score, mean_squared_error],
        "classifier",
        greater_is_better=greater_is_better,
    )
    assert all(entry.greater_is_better == greater_is_better for entry in resolved)


def test_resolve_greater_is_better_invalid():
    """Invalid greater_is_better values raise an informative error."""
    with pytest.raises(ValueError, match="greater_is_better"):
        _resolve_scoring(accuracy_score, "classifier", greater_is_better="yes")


def test_resolve_scorer_style_callable_raises():
    """Scorer style callables are rejected, evaluate cannot use them."""

    def scorer(estimator, X, y):
        return estimator.score(X, y)

    with pytest.raises(TypeError, match="Scorer style callables"):
        _resolve_scoring(scorer, "classifier")


def test_resolve_non_callable_raises():
    """Non-callable, non-string scoring is rejected."""
    with pytest.raises(TypeError, match="scoring must be"):
        _resolve_scoring(42, "classifier")


def test_resolve_empty_raises():
    """Empty scoring collections are rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_scoring([], "classifier")


def test_resolve_duplicate_names_raise():
    """Duplicate metric names are rejected."""
    with pytest.raises(ValueError, match="names must be unique"):
        _resolve_scoring([accuracy_score, accuracy_score], "classifier")


def test_resolve_duplicate_metrics_raise():
    """Distinct names resolving to the same metric function are rejected.

    ``evaluate`` keys its result columns by ``metric.__name__``, so these would
    silently collapse into a single column.
    """
    with pytest.raises(ValueError, match="distinct metric functions"):
        _resolve_scoring(["f1_macro", "f1_micro"], "classifier")


def test_resolve_invalid_estimator_type():
    """An unknown estimator type is rejected when resolving the default metric."""
    with pytest.raises(ValueError, match="estimator_type"):
        _resolve_scoring(None, "forecaster")


def test_resolve_all_sklearn_scorer_names():
    """All sklearn scorer names resolve, and rank higher-is-better."""
    from sklearn.metrics import get_scorer_names

    for name in get_scorer_names():
        scorer = get_scorer(name)
        if not hasattr(scorer, "_score_func"):
            continue
        (resolved,) = _resolve_scoring(name, "classifier")
        assert resolved.name == name
        assert resolved.sign in (1, -1)
        assert resolved.greater_is_better
