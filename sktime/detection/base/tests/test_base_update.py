"""Tests for BaseDetector.update label pooling."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_appends_not_overwrites():
    """Test that update appends new y events rather than overwriting."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})
    y_new = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_new)

    assert len(d._y) == 3


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_translates_ilocs():
    """Test that update translates batch-local ilocs to pooled positions."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})
    y_new = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_new)

    stored_ilocs = d._y["ilocs"].tolist()

    assert 2 in stored_ilocs
    assert 5 in stored_ilocs
    assert 11 in stored_ilocs
    assert 1 not in stored_ilocs


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_after_fit_without_labels_no_typeerror():
    """Test that update(X, y) succeeds after fit(X, y=None)."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_new = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10])
    d.update(X.iloc[10:], y=y_new)

    assert d._y is not None
    assert len(d._y) == 1


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_none_leaves_y_unchanged():
    """Test that update(X, y=None) leaves self._y unchanged."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(15, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [2, 5]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=None)

    assert len(d._y) == 2
    assert list(d._y["ilocs"]) == [2, 5]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_y_multiple_batches():
    """Test that sequential updates accumulate y correctly."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(30, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [3]})
    y_upd1 = pd.DataFrame({"ilocs": [2]})
    y_upd2 = pd.DataFrame({"ilocs": [4]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:20], y=y_upd1)
    d.update(X.iloc[20:30], y=y_upd2)

    stored_ilocs = sorted(d._y["ilocs"].tolist())
    assert stored_ilocs == [3, 12, 24]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_default_refit_path_appends_y():
    """Test y pooling via default refit path in _update."""
    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.DataFrame({"value": np.arange(20, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [3]})
    y_upd = pd.DataFrame({"ilocs": [2]})

    d = DummyRegularAnomalies().fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_upd)

    stored_ilocs = sorted(d._y["ilocs"].tolist())
    assert stored_ilocs == [3, 12]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_default_refit_path_none_y():
    """Test default refit path when update called with y=None."""
    from sktime.detection.dummy import DummyRegularAnomalies

    X = pd.DataFrame({"value": np.arange(20, dtype=float)})
    y_fit = pd.DataFrame({"ilocs": [3, 7]})

    d = DummyRegularAnomalies().fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=None)

    stored_ilocs = sorted(d._y["ilocs"].tolist())
    assert stored_ilocs == [3, 7]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_interval_ilocs_translated():
    """Test that interval-valued ilocs are correctly shifted."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = pd.DataFrame({"value": np.arange(20, dtype=float)})
    y_fit = pd.DataFrame(
        {
            "ilocs": [
                pd.Interval(1, 3, closed="left"),
                pd.Interval(5, 8, closed="left"),
            ]
        }
    )
    y_upd = pd.DataFrame({"ilocs": [pd.Interval(2, 5, closed="left")]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X.iloc[:10], y=y_fit)
    d.update(X.iloc[10:], y=y_upd)

    stored_ilocs = d._y["ilocs"].tolist()
    assert pd.Interval(1, 3, closed="left") in stored_ilocs
    assert pd.Interval(5, 8, closed="left") in stored_ilocs
    assert pd.Interval(12, 15, closed="left") in stored_ilocs
    assert pd.Interval(2, 5, closed="left") not in stored_ilocs


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_overlapping_batches_correct_translation():
    """Test that overlapping batches produce correctly translated ilocs."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X1 = pd.DataFrame({"value": np.arange(10, dtype=float)})
    X2 = pd.DataFrame(
        {"value": np.arange(5, 15, dtype=float)},
        index=pd.RangeIndex(5, 15),
    )

    y_fit = pd.DataFrame({"ilocs": [2]})
    y_upd = pd.DataFrame({"ilocs": [1]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X1, y=y_fit)
    d.update(X2, y=y_upd)

    stored_ilocs = d._y["ilocs"].tolist()
    assert 2 in stored_ilocs
    assert 6 in stored_ilocs
    assert 11 not in stored_ilocs


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_out_of_order_batches_correct_translation():
    """Test that out-of-order batches correctly re-index events."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X1 = pd.DataFrame(
        {"value": np.arange(10, 20, dtype=float)},
        index=pd.RangeIndex(10, 20),
    )
    X2 = pd.DataFrame(
        {"value": np.arange(0, 10, dtype=float)},
        index=pd.RangeIndex(0, 10),
    )

    y_fit = pd.DataFrame({"ilocs": [2]})
    y_upd = pd.DataFrame({"ilocs": [3]})

    d = DummyPretrainedAnomalies(strategy="rate").fit(X1, y=y_fit)
    d.update(X2, y=y_upd)

    stored_ilocs = d._y["ilocs"].tolist()
    assert 12 in stored_ilocs
    assert 3 in stored_ilocs
