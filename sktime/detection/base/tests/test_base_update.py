"""Tests for BaseDetector.update label pooling fix.

Regression tests for GitHub issue #11607:
  BaseDetector.update overwrites sparse event labels when pooling batches.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_appends_not_overwrites():
    """Test that update appends new y events rather than overwriting old ones.

    Failure case of bug #11607, issue 1:
    combine_first on RangeIndex causes row 0 of the new batch's y to overwrite
    row 0 of self._y from fit, instead of appending a new event.
    """
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})
    # one event at local iloc 1 within X.iloc[10:],
    # which is pooled iloc 10 + 1 = 11
    y_new = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_new)

    # self._y should have 3 rows: the 2 from fit + the 1 from update
    assert len(d._y) == 3, (
        f"Expected 3 events after update, got {len(d._y)}. "
        f"self._y = {d._y}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_translates_ilocs():
    """Test that update translates batch-local ilocs to pooled-X positions.

    Failure case of bug #11607, issue 2:
    batch-local ilocs are stored without being shifted by len(old_X),
    corrupting event locations when _fit is called on the pooled self._X.
    """
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})
    # event at local iloc 1 in X.iloc[10:], i.e., pooled iloc 10 + 1 = 11
    y_new = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_new)

    stored_ilocs = d._y["ilocs"].tolist()

    # The events from fit (ilocs 2 and 5) must be preserved
    assert 2 in stored_ilocs, (
        f"Event at iloc 2 from fit was lost. stored ilocs: {stored_ilocs}"
    )
    assert 5 in stored_ilocs, (
        f"Event at iloc 5 from fit was lost. stored ilocs: {stored_ilocs}"
    )

    # The new event's iloc must be shifted: local 1 -> pooled 11 (10 + 1)
    assert 11 in stored_ilocs, (
        f"Expected new event iloc 11 (= 10 + 1) in pooled self._y, "
        f"but got: {stored_ilocs}"
    )

    # The unshifted local iloc (1) must NOT appear
    assert 1 not in stored_ilocs, (
        f"Batch-local iloc 1 was stored un-shifted in self._y: {stored_ilocs}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_after_fit_without_labels_no_typeerror():
    """Test that update(X, y) does not raise TypeError after fit(X, y=None).

    Failure case of bug #11607, issue 3:
    y.combine_first(None) raises TypeError when self._y is None,
    i.e., when fit was called without labels.
    """
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_new = pd.DataFrame({"ilocs": [1]})

    # fit without labels: self._y is None
    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10])

    # this must not raise TypeError
    d.update(X.iloc[10:], y=y_new)

    # after update, self._y should be the shifted y
    assert d._y is not None
    assert len(d._y) == 1


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_none_leaves_y_unchanged():
    """Test that update(X, y=None) does not modify self._y.

    When no labels are supplied to update, self._y should stay as-is.
    """
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=None)

    # self._y should still be the original 2 events from fit
    assert len(d._y) == 2
    assert list(d._y["ilocs"]) == [2, 5]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_multiple_batches():
    """Test that multiple sequential updates accumulate y correctly."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(30, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [3]})  # event at 3 in X[:10]
    y_upd1 = pd.DataFrame({"ilocs": [2]})  # event at 2 in X[10:20], pooled => 12
    y_upd2 = pd.DataFrame({"ilocs": [4]})  # event at 4 in X[20:30], pooled => 24

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:20], y=y_upd1)
    d.update(X.iloc[20:30], y=y_upd2)

    stored_ilocs = sorted(d._y["ilocs"].tolist())

    # expect 3 events: 3 (from fit), 12 (= 10+2, from update 1), 24 (= 20+4, from update 2)
    assert stored_ilocs == [3, 12, 24], (
        f"Expected [3, 12, 24] but got {stored_ilocs}"
    )
