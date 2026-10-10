"""Tests for DummyPatternAnomalies, the detector replaying a pretrained pattern."""

__author__ = ["yash-sangwan"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import DummyPatternAnomalies
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)


def _make_panel_and_events():
    """Make a panel of 2 series with 20 time points each, and 3 known events."""
    X = pd.DataFrame(
        {"value": np.arange(40.0)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], range(20)], names=["instance", "time"]
        ),
    )
    y = pd.DataFrame(
        {"ilocs": [15, 5, 10]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("a", 1), ("b", 0)], names=["instance", "event_no"]
        ),
    )
    return X, y


def _make_series(n_timepoints, start=0, value=None):
    """Make a single series of n_timepoints points."""
    if value is None:
        values = np.arange(float(n_timepoints))
    else:
        values = np.full(n_timepoints, float(value))
    return pd.DataFrame(
        {"value": values}, index=pd.RangeIndex(start, start + n_timepoints)
    )


def test_pretrain_stores_one_pattern_per_series():
    """Test pretrain stores the event positions of every series, from its start."""
    X, y = _make_panel_and_events()

    detector = DummyPatternAnomalies().pretrain(X, y)

    assert detector.state == "pretrained"
    # events are ordered by position, not by the order they appear in y
    assert detector.patterns_ == [(5, 15), (10,)]
    assert detector.get_pretrained_params() == {"patterns_": [(5, 15), (10,)]}


def test_pretrain_without_y_fires_nothing():
    """Test pretrain without known events stores empty patterns."""
    X, _ = _make_panel_and_events()
    X_live = _make_series(30)

    detector = DummyPatternAnomalies().pretrain(X)

    assert detector.patterns_ == [(), ()]
    assert len(detector.fit(X_live).predict(X_live)) == 0


def test_pretrain_flattens_y_instance_labels():
    """Test instance labels of y are flattened the same way as those of X."""
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=10, max_timepoints=10
    )
    y = pd.DataFrame(
        {"ilocs": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples(
            [
                ("h0_0", "h1_0", 0),
                ("h0_0", "h1_1", 0),
                ("h0_1", "h1_0", 0),
                ("nope", "h1_0", 0),
            ],
            names=["h0", "h1", "event_no"],
        ),
    )

    detector = DummyPatternAnomalies().pretrain(X, y)

    # 4 series, the event of the instance not in X is ignored
    assert detector.patterns_ == [(1,), (2,), (3,), ()]


def test_second_pretrain_replaces():
    """Test a second pretrain replaces the patterns of the first."""
    X, y = _make_panel_and_events()
    detector = DummyPatternAnomalies().pretrain(X, y)
    assert detector.patterns_ == [(5, 15), (10,)]

    detector.pretrain(X, y.iloc[:1])

    assert detector.state == "pretrained"
    assert detector.patterns_ == [(15,), ()]


def test_fit_picks_a_stored_pattern_at_random():
    """Test fit picks one of the stored patterns, and not always the same one."""
    X, y = _make_panel_and_events()
    X_live = _make_series(30)

    picked = []
    for random_state in range(6):
        detector = DummyPatternAnomalies(random_state=random_state)
        detector.pretrain(X, y).fit(X_live)
        assert detector.pattern_ in detector.patterns_
        picked.append(detector.pattern_)

    # the pick is random, so both stored patterns are seen over these seeds
    assert set(picked) == {(5, 15), (10,)}


def test_predict_replays_the_picked_pattern():
    """Test predict fires exactly at the positions of the picked pattern."""
    X, y = _make_panel_and_events()
    X_live = _make_series(30)

    detector = DummyPatternAnomalies(random_state=42).pretrain(X, y).fit(X_live)

    assert detector.pattern_ == (5, 15)
    assert list(detector.predict(X_live)["ilocs"]) == [5, 15]


def test_predict_ignores_the_values_and_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed, and do not depend on its values."""
    X, y = _make_panel_and_events()
    detector = DummyPatternAnomalies(random_state=42).pretrain(X, y)
    detector.fit(_make_series(100))

    y_pred = detector.predict(_make_series(30, start=100, value=7))

    assert list(y_pred["ilocs"]) == [5, 15]


def test_predict_drops_alarms_after_the_end():
    """Test alarms beyond the end of the series passed are not fired."""
    X, y = _make_panel_and_events()
    detector = DummyPatternAnomalies(random_state=42).pretrain(X, y)
    detector.fit(_make_series(30))

    # the pattern is (5, 15), so only the first alarm fits into 10 points
    assert list(detector.predict(_make_series(10))["ilocs"]) == [5]


def test_fit_without_pretrain_and_without_y_fires_nothing():
    """Test a detector that saw no events at all fires nothing."""
    X = _make_series(30)

    detector = DummyPatternAnomalies().fit(X)
    y_pred = detector.predict(X)

    assert detector.pattern_ == ()
    assert len(y_pred) == 0
    assert list(y_pred.columns) == ["ilocs"]


def test_fit_without_pretrain_uses_y():
    """Test the pattern is taken from the fit series if there was no pretrain."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [11, 3]})

    detector = DummyPatternAnomalies().fit(X, y)

    assert detector.pattern_ == (3, 11)
    assert list(detector.predict(_make_series(30))["ilocs"]) == [3, 11]


def test_pretrained_patterns_win_over_fit_y():
    """Test a pretrained pattern is used, even if fit also gets known events."""
    X, y = _make_panel_and_events()
    detector = DummyPatternAnomalies(random_state=42).pretrain(X, y)

    detector.fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3]}))

    assert detector.pattern_ == (5, 15)


def test_clone_keeps_patterns():
    """Test a clone of a pretrained detector keeps the patterns, and can be fitted."""
    X, y = _make_panel_and_events()
    detector = DummyPatternAnomalies(random_state=42).pretrain(X, y)

    detector_clone = detector.clone()

    assert detector_clone.state == "pretrained"
    assert detector_clone.get_pretrained_params() == detector.get_pretrained_params()

    X_live = _make_series(30)
    assert list(detector_clone.fit(X_live).predict(X_live)["ilocs"]) == [5, 15]
