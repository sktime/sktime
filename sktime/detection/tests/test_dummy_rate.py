"""Tests for DummyRateAnomalies, the detector firing at the pretrained event rate."""

__author__ = ["yash-sangwan"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import DummyRateAnomalies
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
        {"ilocs": [5, 15, 10]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("a", 1), ("b", 0)], names=["instance", "event_no"]
        ),
    )
    return X, y


def _make_series(n_timepoints, start=0):
    """Make a single series of n_timepoints points."""
    return pd.DataFrame(
        {"value": np.arange(float(n_timepoints))},
        index=pd.RangeIndex(start, start + n_timepoints),
    )


def test_pretrain_learns_event_rate():
    """Test pretrain learns the average number of events per time point."""
    X, y = _make_panel_and_events()

    detector = DummyRateAnomalies().pretrain(X, y)

    assert detector.state == "pretrained"
    assert detector.n_pretrain_events_ == 3
    assert detector.n_pretrain_timepoints_ == 40
    assert detector.pretrain_event_rate_ == 3 / 40
    assert detector.get_pretrained_params() == {
        "n_pretrain_events_": 3,
        "n_pretrain_timepoints_": 40,
        "pretrain_event_rate_": 3 / 40,
    }


def test_pretrain_without_y_fires_nothing():
    """Test pretrain without known events learns a zero rate."""
    X, _ = _make_panel_and_events()
    X_live = _make_series(30)

    detector = DummyRateAnomalies().pretrain(X)

    assert detector.pretrain_event_rate_ == 0
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

    detector = DummyRateAnomalies().pretrain(X, y)

    # 4 series of 10 points, the event of the instance not in X is ignored
    assert detector.n_pretrain_timepoints_ == 40
    assert detector.n_pretrain_events_ == 3
    assert detector.pretrain_event_rate_ == 3 / 40


def test_second_pretrain_replaces():
    """Test a second pretrain replaces the rate of the first."""
    X, y = _make_panel_and_events()
    detector = DummyRateAnomalies().pretrain(X, y)
    assert detector.n_pretrain_events_ == 3

    detector.pretrain(X, y.iloc[:1])

    assert detector.state == "pretrained"
    assert detector.n_pretrain_events_ == 1
    assert detector.pretrain_event_rate_ == 1 / 40


@pytest.mark.parametrize(
    "n_timepoints, expected", [(30, [9, 19, 29]), (12, [9]), (5, [])]
)
def test_predict_fires_at_pretrained_rate(n_timepoints, expected):
    """Test alarms come every round(1 / rate) points, on the series passed."""
    X, _ = _make_panel_and_events()
    # 4 events in 40 time points, so the rate is 0.1, and the step is 10
    y = pd.DataFrame(
        {"ilocs": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)], names=["instance", "event_no"]
        ),
    )
    detector = DummyRateAnomalies().pretrain(X, y)

    X_live = _make_series(n_timepoints)
    y_pred = detector.fit(X_live).predict(X_live)

    assert list(y_pred["ilocs"]) == expected


def test_predict_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed to predict, not on the fit series."""
    X, y = _make_panel_and_events()
    detector = DummyRateAnomalies().pretrain(X, y)
    detector.fit(_make_series(100))

    # rate 3 / 40, so the step is round(40 / 3) = 13
    y_pred = detector.predict(_make_series(30, start=100))

    assert list(y_pred["ilocs"]) == [12, 25]


def test_fit_without_pretrain_and_without_y_fires_nothing():
    """Test a detector that saw no events at all fires nothing."""
    X = _make_series(30)

    detector = DummyRateAnomalies().fit(X)
    y_pred = detector.predict(X)

    assert detector.event_rate_ == 0
    assert len(y_pred) == 0
    assert list(y_pred.columns) == ["ilocs"]


def test_fit_without_pretrain_uses_y():
    """Test the rate is learnt from the fit series if there was no pretrain."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [3, 11]})

    detector = DummyRateAnomalies().fit(X, y)

    assert detector.event_rate_ == 0.1
    assert list(detector.predict(_make_series(30))["ilocs"]) == [9, 19, 29]


def test_pretrained_rate_wins_over_fit_y():
    """Test the pretrained rate is used, even if fit also gets known events."""
    X, y = _make_panel_and_events()
    detector = DummyRateAnomalies().pretrain(X, y)

    detector.fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3, 4, 5]}))

    assert detector.event_rate_ == 3 / 40


def test_clone_keeps_pretrained_rate():
    """Test a clone of a pretrained detector keeps the rate, and can be fitted."""
    X, y = _make_panel_and_events()
    detector = DummyRateAnomalies().pretrain(X, y)

    detector_clone = detector.clone()

    assert detector_clone.state == "pretrained"
    assert detector_clone.get_pretrained_params() == detector.get_pretrained_params()

    X_live = _make_series(30)
    assert list(detector_clone.fit(X_live).predict(X_live)["ilocs"]) == [12, 25]
