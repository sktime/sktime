"""Tests for DummyTimeFreqAnomalies, firing at per-time-from-start rates."""

__author__ = ["yash-sangwan"]

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import DummyTimeFreqAnomalies
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.deep_equals import deep_equals

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)


def _make_panel(lengths):
    """Make a panel with one series per entry of lengths, named a, b, c, ..."""
    names = [chr(ord("a") + i) for i in range(len(lengths))]
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name, n in zip(names, lengths) for t in range(n)],
        names=["instance", "time"],
    )
    return pd.DataFrame({"value": np.arange(float(len(index)))}, index=index)


def _make_events(pairs):
    """Make locked y from (instance, time-from-start) pairs."""
    event_no = {}
    index = []
    for instance, _ in pairs:
        index.append((instance, event_no.get(instance, 0)))
        event_no[instance] = event_no.get(instance, 0) + 1
    return pd.DataFrame(
        {"ilocs": [offset for _, offset in pairs]},
        index=pd.MultiIndex.from_tuples(index, names=["instance", "event_no"]),
    )


def _make_series(n_timepoints, start=0, value=None):
    """Make a single series of n_timepoints points."""
    if value is None:
        values = np.arange(float(n_timepoints))
    else:
        values = np.full(n_timepoints, float(value))
    return pd.DataFrame(
        {"value": values}, index=pd.RangeIndex(start, start + n_timepoints)
    )


def test_pretrain_counts_events_per_time_from_start():
    """Test pretrain counts events per time-from-start, over the series at risk."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5), ("b", 10)])

    detector = DummyTimeFreqAnomalies().pretrain(X, y)

    assert detector.state == "pretrained"
    assert len(detector.event_prob_) == 20
    assert detector.event_counts_[5] == 2
    assert detector.event_counts_[10] == 1
    assert detector.event_counts_.sum() == 3
    assert list(detector.n_at_risk_) == [2] * 20
    assert detector.event_prob_[5] == 1.0
    assert detector.event_prob_[10] == 0.5
    assert detector.event_prob_[0] == 0.0
    assert sorted(detector.get_pretrained_params()) == [
        "event_counts_",
        "event_prob_",
        "n_at_risk_",
    ]


def test_short_series_does_not_shrink_late_probability():
    """Test the denominator is the series at risk, not the total series count."""
    X = _make_panel([20, 5])
    y = _make_events([("a", 10)])

    detector = DummyTimeFreqAnomalies().pretrain(X, y)

    # both series reach time-from-start 0, only the long one reaches 10
    assert detector.n_at_risk_[0] == 2
    assert detector.n_at_risk_[10] == 1
    assert detector.event_prob_[10] == 1.0


def test_event_past_the_end_of_its_series_is_ignored():
    """Test an event after the end of its own series does not raise the count."""
    X = _make_panel([20, 5])
    y = _make_events([("a", 10), ("b", 10)])

    detector = DummyTimeFreqAnomalies().pretrain(X, y)

    # only the 20 point series reaches time-from-start 10, so only its event counts
    assert detector.n_at_risk_[10] == 1
    assert detector.event_counts_[10] == 1
    assert detector.event_prob_[10] == 1.0
    assert detector.event_prob_.max() <= 1.0


def test_pretrain_without_y_fires_nothing():
    """Test pretrain without known events learns zero probabilities."""
    X = _make_panel([20, 20])
    X_live = _make_series(20)

    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X)

    assert detector.event_counts_.sum() == 0
    assert not detector.event_prob_.any()
    assert len(detector.fit(X_live).predict(X_live)) == 0


def test_pretrain_flattens_y_instance_labels():
    """Test instance labels of y are flattened the same way as those of X."""
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=10, max_timepoints=10
    )
    y = pd.DataFrame(
        {"ilocs": [3, 3, 3, 7]},
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

    detector = DummyTimeFreqAnomalies().pretrain(X, y)

    # 4 series of 10 points, the event of the instance not in X is ignored
    assert list(detector.n_at_risk_) == [4] * 10
    assert detector.event_counts_[3] == 3
    assert detector.event_counts_[7] == 0
    assert detector.event_prob_[3] == 0.75


def test_second_pretrain_replaces():
    """Test a second pretrain replaces the counts of the first."""
    X = _make_panel([20, 20])
    detector = DummyTimeFreqAnomalies().pretrain(X, _make_events([("a", 5), ("b", 5)]))
    assert detector.event_prob_[5] == 1.0

    detector.pretrain(X, _make_events([("a", 12)]))

    assert detector.state == "pretrained"
    assert detector.event_prob_[5] == 0.0
    assert detector.event_prob_[12] == 0.5


@pytest.mark.parametrize("random_state", [0, 1, 2, 3])
def test_probability_one_always_fires_and_zero_never(random_state):
    """Test a time-from-start of probability one fires, and zero does not."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])
    X_live = _make_series(20)

    detector = DummyTimeFreqAnomalies(random_state=random_state).pretrain(X, y)
    y_pred = detector.fit(X_live).predict(X_live)

    assert list(y_pred["ilocs"]) == [5]


def test_predict_ignores_the_values_and_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed, and do not depend on its values."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])

    detector = DummyTimeFreqAnomalies(random_state=1).pretrain(X, y)
    detector.fit(_make_series(100))

    y_pred = detector.predict(_make_series(20, start=100, value=7))

    assert list(y_pred["ilocs"]) == [5]


def test_predict_fires_nothing_past_the_learnt_horizon():
    """Test no alarm is fired beyond the last time-from-start seen in pretrain."""
    X = _make_panel([5, 5])
    # every time-from-start of both series has an event, so the curve is all ones
    y = _make_events([(name, t) for name in ["a", "b"] for t in range(5)])

    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X, y)
    detector.fit(_make_series(12))

    y_pred = detector.predict(_make_series(12))

    assert list(y_pred["ilocs"]) == [0, 1, 2, 3, 4]


def test_same_seed_gives_the_same_alarms():
    """Test a fixed random_state fires the same alarms on the same length."""
    X = _make_panel([20, 20])
    # only one of the two series has an event at each time-from-start
    y = _make_events([("a", t) for t in range(20)])
    X_live = _make_series(20)

    def alarms(random_state):
        detector = DummyTimeFreqAnomalies(random_state=random_state)
        detector.pretrain(X, y).fit(X_live)
        return list(detector.predict(X_live)["ilocs"])

    # one of the two series has an event at every time-from-start
    prob = DummyTimeFreqAnomalies().pretrain(X, y).event_prob_
    assert (prob == 0.5).all()

    assert alarms(0) == alarms(0)
    assert alarms(0) != alarms(1)


def test_predict_does_not_change_the_detector():
    """Test predict draws without writing anything to the detector."""
    X = _make_panel([20, 20])
    y = _make_events([("a", t) for t in range(20)])
    X_live = _make_series(20)

    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X, y).fit(X_live)
    prob_before = detector.prob_.copy()
    keys_before = set(vars(detector))

    detector.predict(X_live)

    assert set(vars(detector)) == keys_before
    assert np.array_equal(detector.prob_, prob_before)


def test_fit_without_pretrain_and_without_y_fires_nothing():
    """Test a detector that saw no events at all fires nothing."""
    X = _make_series(20)

    detector = DummyTimeFreqAnomalies(random_state=0).fit(X)
    y_pred = detector.predict(X)

    assert len(detector.prob_) == 0
    assert len(y_pred) == 0
    assert list(y_pred.columns) == ["ilocs"]


@pytest.mark.parametrize("random_state", [0, 1])
def test_fit_without_pretrain_uses_y(random_state):
    """Test probability one at the event offsets of the fit series."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [3, 11]})

    detector = DummyTimeFreqAnomalies(random_state=random_state).fit(X, y)

    assert list(np.flatnonzero(detector.prob_ == 1.0)) == [3, 11]
    assert list(detector.predict(X)["ilocs"]) == [3, 11]


def test_pretrained_probabilities_win_over_fit_y():
    """Test the pretrained curve is used, even if fit also gets known events."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])

    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X, y)
    detector.fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3]}))

    assert np.array_equal(detector.prob_, detector.event_prob_)
    assert list(detector.predict(_make_series(20))["ilocs"]) == [5]


def test_clone_keeps_pretrained_probabilities():
    """Test a clone of a pretrained detector keeps the curve, and can be fitted."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])
    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X, y)

    detector_clone = detector.clone()

    assert detector_clone.state == "pretrained"
    assert np.array_equal(detector_clone.event_prob_, detector.event_prob_)

    X_live = _make_series(20)
    assert list(detector_clone.fit(X_live).predict(X_live)["ilocs"]) == [5]


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


def _half_prob_detector(random_state=0):
    """Make a detector with probability 0.5 at every time-from-start 0 to 19."""
    X = _make_panel([20, 20])
    y = _make_events([("a", t) for t in range(20)])
    return DummyTimeFreqAnomalies(random_state=random_state).pretrain(X, y)


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_chunked_replay_equals_one_shot_predict(chunk_size):
    """Test update_predict on chunks fires where predict on the whole series does."""
    X = _make_series(20)

    expected = list(_half_prob_detector().fit(X).predict(X)["ilocs"])
    times = _replay(_half_prob_detector(), X, warmup=2, chunk_size=chunk_size)

    # the one-shot alarms are a real random draw, neither all nor none
    assert 0 < len(expected) < 20
    assert times == [t for t in expected if t >= 2]


def test_same_seed_same_cursor_same_length_same_alarms():
    """Test the same seed, cursor and length of X give the same alarms."""
    X = _make_series(20)
    first = _half_prob_detector(3).fit(X.iloc[:4]).update(X.iloc[4:14])
    second = _half_prob_detector(3).fit(X.iloc[:4]).update(X.iloc[4:14])
    assert first.cursor_ == second.cursor_ == 4

    alarms = list(first.predict(X.iloc[4:14])["ilocs"])

    assert alarms == list(second.predict(X.iloc[4:14])["ilocs"])
    assert alarms == list(first.predict(X.iloc[4:14])["ilocs"])


def test_ilocs_stay_inside_the_chunk():
    """Test predict after update returns ilocs on the chunk, not on the stream."""
    X_pretrain = _make_panel([20, 20])
    y_pretrain = _make_events([("a", 5), ("b", 5)])
    X = _make_series(20)
    detector = DummyTimeFreqAnomalies(random_state=0).pretrain(X_pretrain, y_pretrain)
    detector.fit(X.iloc[:4])

    # the chunk holds times-from-start 4 to 9, the event at 5 is iloc 1
    assert list(detector.update_predict(X.iloc[4:10])["ilocs"]) == [1]
    # the next chunk holds 10 to 19, where the probability is 0
    assert len(detector.update_predict(X.iloc[10:20])) == 0


def test_predict_after_update_does_not_change_the_detector():
    """Test predict leaves the detector unchanged, also after update."""
    X = _make_series(20)
    detector = _half_prob_detector().fit(X.iloc[:4]).update(X.iloc[4:14])
    before = deepcopy(vars(detector))

    detector.predict(X.iloc[4:14])

    assert deep_equals(vars(detector), before)


def test_second_fit_resets_the_cursor():
    """Test fit starts the stream again at time-from-start 0."""
    X = _make_series(20)
    detector = _half_prob_detector().fit(X.iloc[:4])
    detector.update(X.iloc[4:10]).update(X.iloc[10:16])
    assert detector.cursor_ == 10
    assert detector.n_timepoints_seen_ == 16

    detector.fit(X)

    assert detector.cursor_ == 0
    assert detector.n_timepoints_seen_ == 20
    fresh = _half_prob_detector().fit(X)
    assert list(detector.predict(X)["ilocs"]) == list(fresh.predict(X)["ilocs"])
