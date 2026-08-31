"""Tests for the SyncToLongest splitter composition."""

__author__ = ["Garve"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sktime.split import SameLocSplitter, SlidingWindowSplitter, TestPlusTrainSplitter
from sktime.split.compose import SyncToLongest
from sktime.utils._testing.series import _make_series


def _panel(lengths: dict) -> pd.DataFrame:
    """Build a panel where each instance is a suffix of a shared 0..9 timeline."""
    series = {
        name: pd.Series(np.arange(start, 10), index=pd.RangeIndex(start, 10))
        for name, start in lengths.items()
    }
    return pd.concat(series, names=["instance", "time"]).to_frame("value")


def test_sync_to_longest_excludes_instances_without_full_window():
    """Folds should only contain instances that have started by the cutoff.

    This is the scenario from the feature request: an early fold, before the
    shorter instance has enough history, should contain only the longer
    instance; later folds, once the shorter instance has caught up, should
    contain both.
    """
    y = _panel({"long": 0, "short": 3})
    cv = SyncToLongest(
        SlidingWindowSplitter(window_length=3, fh=1, step_length=3)
    )

    splits = list(cv.split_series(y))
    assert len(splits) == 3

    instances_per_fold = [set(train.index.get_level_values(0)) for train, _ in splits]
    assert instances_per_fold[0] == {"long"}
    assert instances_per_fold[1] == {"long", "short"}
    assert instances_per_fold[2] == {"long", "short"}

    # the shared fold uses each instance's own train/test window
    train1, test1 = splits[1]
    assert list(train1.loc["long"].index) == [3, 4, 5]
    assert list(test1.loc["long"].index) == [6]
    assert list(train1.loc["short"].index) == [3, 4, 5]
    assert list(test1.loc["short"].index) == [6]


def test_sync_to_longest_min_length_relaxes_training_window():
    """min_length should include instances with a partial training window."""
    y = _panel({"long": 0, "medium": 2, "short": 3})
    base_cv = SlidingWindowSplitter(window_length=3, fh=1, step_length=3)

    cv_strict = SyncToLongest(base_cv)
    first_fold_instances = set(
        next(cv_strict.split_series(y))[0].index.get_level_values(0)
    )
    assert "medium" not in first_fold_instances

    cv_relaxed = SyncToLongest(base_cv, min_length=1)
    train0, _ = next(cv_relaxed.split_series(y))
    assert "medium" in set(train0.index.get_level_values(0))
    assert list(train0.loc["medium"].index) == [2]  # only 1 point available


def test_sync_to_longest_forwards_fh_and_window_length():
    """cv.fh/cv.window_length must match base_cv's, not BaseSplitter's defaults.

    evaluate() reads cv.fh (not base_cv.fh) to fit/predict each fold. If it
    silently fell back to the BaseSplitter default of fh=1, a multi-step
    base_cv.fh would make evaluate() fit/predict the wrong number of steps,
    while y_test still has the real (longer) horizon -- causing a shape
    mismatch between y_test and y_pred deep inside metric computation.
    """
    base_cv = SlidingWindowSplitter(window_length=3, fh=[1, 2], step_length=3)
    cv = SyncToLongest(base_cv)
    assert cv.fh == base_cv.fh
    assert cv.window_length == base_cv.window_length

    y = _panel({"long": 0, "short": 3})
    results = evaluate(forecaster=NaiveForecaster(), y=y, cv=cv)
    assert len(results) == cv.get_n_splits(y)


def test_sync_to_longest_invalid_min_length_raises():
    with pytest.raises(ValueError, match="min_length"):
        SyncToLongest(SlidingWindowSplitter(), min_length=0)


def test_sync_to_longest_min_length_above_window_warns():
    """min_length above window_length has no relaxing effect, so it should warn."""
    with pytest.warns(UserWarning, match="min_length"):
        SyncToLongest(
            SlidingWindowSplitter(window_length=3, fh=1), min_length=5
        )


def test_sync_to_longest_works_with_evaluate_and_exogenous_X():
    """evaluate() should run end to end when X is passed alongside y."""
    y = _panel({"long": 0, "short": 3})
    X = y.rename(columns={"value": "exog"}) * 10
    cv = SyncToLongest(
        SlidingWindowSplitter(window_length=3, fh=1, step_length=3)
    )
    forecaster = NaiveForecaster()

    results = evaluate(forecaster=forecaster, y=y, X=X, cv=cv)

    assert len(results) == cv.get_n_splits(y)
    assert results["test_MeanAbsolutePercentageError"].notna().all()


def test_sync_to_longest_with_explicit_cv_x():
    """evaluate() should accept an explicit, different cv_X for X.

    cv_X wraps SyncToLongest in TestPlusTrainSplitter, so X_test
    covers the train+test region instead of just the test region. It is
    further wrapped in SameLocSplitter(..., y) -- the same composition
    evaluate() itself uses by default -- so that instance qualification for
    each fold is always decided from y, not independently from X. Without
    that wrapping, an X whose per-instance coverage differs from y's (see
    test_sync_to_longest_explicit_cv_x_needs_same_loc below) makes
    SyncToLongest pick a different set of instances for X than for
    y, silently breaking downstream shapes.
    """
    y = _panel({"long": 0, "short": 3})
    X = y.rename(columns={"value": "exog"}) * 10
    base_cv = SlidingWindowSplitter(window_length=3, fh=1, step_length=3)
    cv = SyncToLongest(base_cv)
    cv_X = SameLocSplitter(TestPlusTrainSplitter(SyncToLongest(base_cv)), y)

    results = evaluate(forecaster=NaiveForecaster(), y=y, X=X, cv=cv, cv_X=cv_X)

    assert len(results) == cv.get_n_splits(y)
    assert results["test_MeanAbsolutePercentageError"].notna().all()

    # cv_X's test fold is the union of cv's train and test fold
    y_train, y_test = next(cv.split_series(y))
    X_train, X_test = next(cv_X.split_series(X))
    assert set(X_train.index) == set(y_train.index)
    assert set(X_test.index) == set(y_train.index) | set(y_test.index)


def test_sync_to_longest_explicit_cv_x_needs_same_loc():
    """An explicit cv_X built on SyncToLongest must wrap SameLocSplitter.

    If X's per-instance coverage differs from y's (here "short" ends earlier
    in X than in y), an unwrapped cv_X re-decides instance qualification
    from X's own coverage, independently of y, and picks a different
    instance set for a fold than cv did for y -- so evaluate() fails deep
    inside forecaster/metric code with a shape mismatch that gives no hint
    the real problem is in the splitter composition.

    Wrapping cv_X in SameLocSplitter(..., y) ties X's fold selection to y's,
    so evaluate() now fails immediately and clearly: a KeyError naming the
    (instance, time) pair y needs that X doesn't have.
    """
    y = _panel({"long": 0, "short": 3})  # "short" covers 3..9
    X = _panel({"long": 0, "short": 5}).rename(columns={"value": "exog"}) * 10
    base_cv = SlidingWindowSplitter(window_length=3, fh=1, step_length=1)
    cv = SyncToLongest(base_cv)

    unsafe_cv_X = TestPlusTrainSplitter(SyncToLongest(base_cv))
    safe_cv_X = SameLocSplitter(unsafe_cv_X, y)

    # unwrapped: X is split independently, "short" silently drops out of
    # X_test earlier (once X's coverage ends) than out of y_test
    y_test_insts = [set(t.index.get_level_values(0)) for _, t in cv.split_series(y)]
    x_test_insts = [
        set(t.index.get_level_values(0)) for _, t in unsafe_cv_X.split_series(X)
    ]
    assert y_test_insts != x_test_insts

    # wrapped: fails fast and clearly instead of deep inside evaluate()
    with pytest.raises(KeyError, match="short"):
        list(safe_cv_X.split_series(X))


def test_sync_to_longest_works_with_mase():
    """evaluate() should run with MeanAbsoluteScaledError, which needs y_train."""
    y = _panel({"long": 0, "short": 3})
    cv = SyncToLongest(
        SlidingWindowSplitter(window_length=3, fh=1, step_length=3)
    )
    forecaster = NaiveForecaster()
    scoring = MeanAbsoluteScaledError()

    results = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=scoring)

    assert len(results) == cv.get_n_splits(y)
    assert results[f"test_{scoring.name}"].notna().all()


def test_sync_to_longest_passthrough_for_single_series():
    """SyncToLongest on a plain (non-panel) series should be a no-op."""
    y = _make_series(n_timepoints=20)
    base_cv = SlidingWindowSplitter(window_length=5, fh=1)
    cv = SyncToLongest(base_cv)

    expected = list(base_cv.split(y))
    actual = list(cv.split(y))

    assert len(actual) == len(expected)
    for (train, test), (expected_train, expected_test) in zip(actual, expected):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
