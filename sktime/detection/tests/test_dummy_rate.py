"""Tests for DummyRateAnomalies, the detector firing at the pretrained event rate."""

__author__ = ["yash-sangwan"]

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import DummyRateAnomalies
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.deep_equals import deep_equals

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


def _replay(detector, X, warmup, chunk_size):
    """Fit on the first warmup points, then update_predict the rest in chunks.

    Returns the alarms as times-from-start of X, and checks that every iloc
    returned by update_predict is inside its chunk.
    """
    detector.fit(X.iloc[:warmup])
    times = []
    for start in range(warmup, len(X), chunk_size):
        chunk = X.iloc[start : start + chunk_size]
        ilocs = list(detector.update_predict(chunk)["ilocs"])
        assert all(0 <= iloc < len(chunk) for iloc in ilocs)
        times += [start + iloc for iloc in ilocs]
    return times


def _pretrained():
    """Make a detector pretrained on the panel, at rate 3 / 40, so step 13."""
    X, y = _make_panel_and_events()
    return DummyRateAnomalies().pretrain(X, y)


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_chunked_replay_equals_one_shot_predict(chunk_size):
    """Test update_predict on chunks fires where predict on the whole series does."""
    X = _make_series(40)

    expected = list(_pretrained().fit(X).predict(X)["ilocs"])
    times = _replay(_pretrained(), X, warmup=4, chunk_size=chunk_size)

    # step 13, so the alarms are at time-from-start 12, 25 and 38
    assert expected == [12, 25, 38]
    assert times == expected


def test_ilocs_stay_inside_the_chunk():
    """Test predict after update returns ilocs on the chunk, not on the stream."""
    X = _make_series(40)
    detector = _pretrained().fit(X.iloc[:4])

    # the chunk holds times-from-start 4 to 15, the alarm at 12 is iloc 8
    assert list(detector.update_predict(X.iloc[4:16])["ilocs"]) == [8]
    # the next chunk holds 16 to 29, the alarm at 25 is iloc 9
    assert list(detector.update_predict(X.iloc[16:30])["ilocs"]) == [9]


def test_predict_after_update_does_not_change_the_detector():
    """Test predict leaves the detector unchanged, also after update."""
    X = _make_series(40)
    detector = _pretrained().fit(X.iloc[:4]).update(X.iloc[4:16])
    before = deepcopy(vars(detector))

    detector.predict(X.iloc[4:16])

    assert deep_equals(vars(detector), before)


def test_second_fit_resets_the_cursor():
    """Test fit starts the stream again at time-from-start 0."""
    X = _make_series(40)
    detector = _pretrained().fit(X.iloc[:4])
    detector.update(X.iloc[4:16]).update(X.iloc[16:30])
    assert detector.cursor_ == 16
    assert detector.n_timepoints_seen_ == 30

    detector.fit(X)

    assert detector.cursor_ == 0
    assert detector.n_timepoints_seen_ == 40
    assert list(detector.predict(X)["ilocs"]) == [12, 25, 38]
