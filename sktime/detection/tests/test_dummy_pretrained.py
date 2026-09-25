"""Tests for DummyPretrainedAnomalies, the dummy detector with pretrain strategies."""

__author__ = ["yash-sangwan"]

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sktime.detection.dummy import DummyPretrainedAnomalies
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.deep_equals import deep_equals

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)

STRATEGIES = ["rate", "pattern", "timefreq"]

# attributes written by pretrain, per strategy
PRETRAINED_PARAMS = {
    "rate": ["n_pretrain_events_", "n_pretrain_timepoints_", "pretrain_event_rate_"],
    "pattern": ["patterns_"],
    "timefreq": ["event_counts_", "event_prob_", "n_at_risk_"],
}

# attribute written by fit and used by predict, per strategy
FIT_ATTRIBUTE = {"rate": "event_rate_", "pattern": "pattern_", "timefreq": "prob_"}


def _make_panel(lengths):
    """Make a panel with one series per entry of lengths, named a, b, c, ..."""
    names = [chr(ord("a") + i) for i in range(len(lengths))]
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name, n in zip(names, lengths) for t in range(n)],
        names=["instance", "time"],
    )
    return pd.DataFrame({"value": np.arange(float(len(index)))}, index=index)


def _make_events(pairs):
    """Make y in the pretrain format, from (instance, time-from-start) pairs."""
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


def _panel_and_events():
    """Make a panel of 2 series of 20 points, with 3 known events.

    Series a has events at 15 and 5, in that order in y, and series b at 10.
    """
    return _make_panel([20, 20]), _make_events([("a", 15), ("a", 5), ("b", 10)])


def _hierarchical_panel_and_events(ilocs):
    """Make 4 series of 10 points in a 2-level hierarchy, and 4 events.

    The first 3 events belong to the first 3 series. The 4th event belongs to
    an instance that is not in X, so the 4th series has no event.
    """
    X = _make_hierarchical(
        hierarchy_levels=(2, 2), min_timepoints=10, max_timepoints=10
    )
    y = pd.DataFrame(
        {"ilocs": ilocs},
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
    return X, y


def _pretrained(strategy, random_state=None):
    """Make a pretrained detector, with known alarms on a fresh series.

    * ``"rate"``: 3 events in 40 points, so step 13, alarms at 12, 25, 38, ...
    * ``"pattern"``: random_state 42 picks the pattern (5, 15)
    * ``"timefreq"``: probability 0.5 at every time-from-start 0 to 19
    """
    if strategy == "timefreq":
        X = _make_panel([20, 20])
        y = _make_events([("a", t) for t in range(20)])
        random_state = 0 if random_state is None else random_state
    else:
        X, y = _panel_and_events()
        random_state = 42 if random_state is None else random_state
    detector = DummyPretrainedAnomalies(strategy=strategy, random_state=random_state)
    return detector.pretrain(X, y)


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


def _alarms(detector, X):
    """Return the alarms of predict on X, as a list of int."""
    return list(detector.predict(X)["ilocs"])


# strategy argument
# -----------------


def test_default_strategy_is_rate():
    """Test the default strategy is rate, and learns the rate in pretrain."""
    X, y = _panel_and_events()

    detector = DummyPretrainedAnomalies()

    assert detector.strategy == "rate"
    assert detector.pretrain(X, y).pretrain_event_rate_ == 3 / 40


@pytest.mark.parametrize("strategy", ["nope", "Rate", None])
def test_invalid_strategy_raises_in_pretrain(strategy):
    """Test pretrain raises a ValueError for an unknown strategy."""
    X, y = _panel_and_events()

    with pytest.raises(ValueError, match="strategy"):
        DummyPretrainedAnomalies(strategy=strategy).pretrain(X, y)


@pytest.mark.parametrize("strategy", ["nope", "Rate", None])
def test_invalid_strategy_raises_in_fit(strategy):
    """Test fit raises a ValueError for an unknown strategy, without pretrain."""
    with pytest.raises(ValueError, match="strategy"):
        DummyPretrainedAnomalies(strategy=strategy).fit(_make_series(20))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pretrain_writes_only_the_attributes_of_its_strategy(strategy):
    """Test pretrain writes the attributes of the chosen strategy, and no others."""
    detector = _pretrained(strategy)

    assert detector.state == "pretrained"
    assert sorted(detector.get_pretrained_params()) == PRETRAINED_PARAMS[strategy]

    others = [
        attr
        for other, attrs in PRETRAINED_PARAMS.items()
        if other != strategy
        for attr in attrs
    ]
    assert not any(hasattr(detector, attr) for attr in others)


# contract shared by all strategies
# ---------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pretrain_without_y_fires_nothing(strategy):
    """Test pretrain without known events learns to fire nothing."""
    X, _ = _panel_and_events()
    X_live = _make_series(30)

    detector = DummyPretrainedAnomalies(strategy=strategy, random_state=0)
    detector.pretrain(X).fit(X_live)

    assert len(detector.predict(X_live)) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fit_without_pretrain_and_without_y_fires_nothing(strategy):
    """Test a detector that saw no events at all fires nothing."""
    X = _make_series(30)

    detector = DummyPretrainedAnomalies(strategy=strategy, random_state=0).fit(X)
    y_pred = detector.predict(X)

    assert not np.any(getattr(detector, FIT_ATTRIBUTE[strategy]))
    assert len(y_pred) == 0
    assert list(y_pred.columns) == ["ilocs"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_clone_keeps_pretrained_state(strategy):
    """Test a clone of a pretrained detector keeps what it learnt, and can be fitted."""
    detector = _pretrained(strategy)

    detector_clone = detector.clone()

    assert detector_clone.state == "pretrained"
    assert deep_equals(
        detector_clone.get_pretrained_params(), detector.get_pretrained_params()
    )

    X_live = _make_series(30)
    expected = _alarms(_pretrained(strategy).fit(X_live), X_live)
    assert _alarms(detector_clone.fit(X_live), X_live) == expected


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunked_replay_equals_one_shot_predict(strategy, chunk_size):
    """Test update_predict on chunks fires where predict on the whole series does."""
    X = _make_series(40)
    warmup = 4

    expected = _alarms(_pretrained(strategy).fit(X), X)
    times = _replay(_pretrained(strategy), X, warmup=warmup, chunk_size=chunk_size)

    if strategy == "rate":
        # step 13, so the alarms are at time-from-start 12, 25 and 38
        assert expected == [12, 25, 38]
    elif strategy == "pattern":
        assert expected == [5, 15]
    else:
        # the one-shot alarms are a real random draw, neither all nor none
        assert 0 < len(expected) < 20
    assert times == [t for t in expected if t >= warmup]


@pytest.mark.parametrize(
    "strategy, chunks, expected",
    [
        # step 13: 4 to 15 holds the alarm at 12, 16 to 29 the one at 25
        ("rate", [(4, 16), (16, 30)], [[8], [9]]),
        # pattern (5, 15): 4 to 11 holds 5, 12 to 19 holds 15
        ("pattern", [(4, 12), (12, 20)], [[1], [3]]),
    ],
)
def test_ilocs_stay_inside_the_chunk(strategy, chunks, expected):
    """Test predict after update returns ilocs on the chunk, not on the stream."""
    X = _make_series(40)
    detector = _pretrained(strategy).fit(X.iloc[:4])

    for (start, end), ilocs in zip(chunks, expected):
        assert list(detector.update_predict(X.iloc[start:end])["ilocs"]) == ilocs


def test_ilocs_stay_inside_the_chunk_timefreq():
    """Test predict after update returns ilocs on the chunk, not on the stream."""
    X_pretrain = _make_panel([20, 20])
    y_pretrain = _make_events([("a", 5), ("b", 5)])
    X = _make_series(20)
    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=0)
    detector.pretrain(X_pretrain, y_pretrain).fit(X.iloc[:4])

    # the chunk holds times-from-start 4 to 9, the event at 5 is iloc 1
    assert list(detector.update_predict(X.iloc[4:10])["ilocs"]) == [1]
    # the next chunk holds 10 to 19, where the probability is 0
    assert len(detector.update_predict(X.iloc[10:20])) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_predict_after_update_does_not_change_the_detector(strategy):
    """Test predict leaves the detector unchanged, also after update."""
    X = _make_series(30)
    detector = _pretrained(strategy).fit(X.iloc[:4]).update(X.iloc[4:12])
    before = deepcopy(vars(detector))

    detector.predict(X.iloc[4:12])

    assert deep_equals(vars(detector), before)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_second_fit_resets_the_cursor(strategy):
    """Test fit starts the stream again at time-from-start 0."""
    X = _make_series(30)
    detector = _pretrained(strategy).fit(X.iloc[:4])
    detector.update(X.iloc[4:12]).update(X.iloc[12:20])
    assert detector.cursor_ == 12
    assert detector.n_timepoints_seen_ == 20

    detector.fit(X)

    assert detector.cursor_ == 0
    assert detector.n_timepoints_seen_ == 30
    assert _alarms(detector, X) == _alarms(_pretrained(strategy).fit(X), X)


# strategy "rate"
# ---------------


def test_rate_pretrain_learns_event_rate():
    """Test pretrain learns the average number of events per time point."""
    X, y = _panel_and_events()

    detector = DummyPretrainedAnomalies(strategy="rate").pretrain(X, y)

    assert detector.n_pretrain_events_ == 3
    assert detector.n_pretrain_timepoints_ == 40
    assert detector.pretrain_event_rate_ == 3 / 40
    assert detector.get_pretrained_params() == {
        "n_pretrain_events_": 3,
        "n_pretrain_timepoints_": 40,
        "pretrain_event_rate_": 3 / 40,
    }


def test_rate_pretrain_without_y_learns_zero_rate():
    """Test pretrain without known events learns a zero rate."""
    X, _ = _panel_and_events()

    detector = DummyPretrainedAnomalies(strategy="rate").pretrain(X)

    assert detector.pretrain_event_rate_ == 0


def test_rate_pretrain_flattens_y_instance_labels():
    """Test instance labels of y are flattened the same way as those of X."""
    X, y = _hierarchical_panel_and_events([1, 2, 3, 4])

    detector = DummyPretrainedAnomalies(strategy="rate").pretrain(X, y)

    # 4 series of 10 points, the event of the instance not in X is ignored
    assert detector.n_pretrain_timepoints_ == 40
    assert detector.n_pretrain_events_ == 3
    assert detector.pretrain_event_rate_ == 3 / 40


def test_rate_second_pretrain_replaces():
    """Test a second pretrain replaces the rate of the first."""
    X, y = _panel_and_events()
    detector = DummyPretrainedAnomalies(strategy="rate").pretrain(X, y)
    assert detector.n_pretrain_events_ == 3

    detector.pretrain(X, y.iloc[:1])

    assert detector.state == "pretrained"
    assert detector.n_pretrain_events_ == 1
    assert detector.pretrain_event_rate_ == 1 / 40


@pytest.mark.parametrize(
    "n_timepoints, expected", [(30, [9, 19, 29]), (12, [9]), (5, [])]
)
def test_rate_predict_fires_at_pretrained_rate(n_timepoints, expected):
    """Test alarms come every round(1 / rate) points, on the series passed."""
    X = _make_panel([20, 20])
    # 4 events in 40 time points, so the rate is 0.1, and the step is 10
    y = _make_events([("a", 1), ("a", 2), ("b", 3), ("b", 4)])
    detector = DummyPretrainedAnomalies(strategy="rate").pretrain(X, y)

    X_live = _make_series(n_timepoints)

    assert _alarms(detector.fit(X_live), X_live) == expected


def test_rate_predict_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed to predict, not on the fit series."""
    detector = _pretrained("rate").fit(_make_series(100))

    # rate 3 / 40, so the step is round(40 / 3) = 13
    assert _alarms(detector, _make_series(30, start=100)) == [12, 25]


def test_rate_fit_without_pretrain_uses_y():
    """Test the rate is learnt from the fit series if there was no pretrain."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [3, 11]})

    detector = DummyPretrainedAnomalies(strategy="rate").fit(X, y)

    assert detector.event_rate_ == 0.1
    assert _alarms(detector, _make_series(30)) == [9, 19, 29]


def test_rate_pretrained_rate_wins_over_fit_y():
    """Test the pretrained rate is used, even if fit also gets known events."""
    detector = _pretrained("rate")

    detector.fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3, 4, 5]}))

    assert detector.event_rate_ == 3 / 40


# strategy "pattern"
# ------------------


def test_pattern_pretrain_stores_one_pattern_per_series():
    """Test pretrain stores the event positions of every series, from its start."""
    X, y = _panel_and_events()

    detector = DummyPretrainedAnomalies(strategy="pattern").pretrain(X, y)

    # events are ordered by position, not by the order they appear in y
    assert detector.patterns_ == [(5, 15), (10,)]
    assert detector.get_pretrained_params() == {"patterns_": [(5, 15), (10,)]}


def test_pattern_pretrain_without_y_stores_empty_patterns():
    """Test pretrain without known events stores empty patterns."""
    X, _ = _panel_and_events()

    detector = DummyPretrainedAnomalies(strategy="pattern").pretrain(X)

    assert detector.patterns_ == [(), ()]


def test_pattern_pretrain_flattens_y_instance_labels():
    """Test instance labels of y are flattened the same way as those of X."""
    X, y = _hierarchical_panel_and_events([1, 2, 3, 4])

    detector = DummyPretrainedAnomalies(strategy="pattern").pretrain(X, y)

    # 4 series, the event of the instance not in X is ignored
    assert detector.patterns_ == [(1,), (2,), (3,), ()]


def test_pattern_second_pretrain_replaces():
    """Test a second pretrain replaces the patterns of the first."""
    X, y = _panel_and_events()
    detector = DummyPretrainedAnomalies(strategy="pattern").pretrain(X, y)
    assert detector.patterns_ == [(5, 15), (10,)]

    detector.pretrain(X, y.iloc[:1])

    assert detector.state == "pretrained"
    assert detector.patterns_ == [(15,), ()]


def test_pattern_fit_picks_a_stored_pattern_at_random():
    """Test fit picks one of the stored patterns, and not always the same one."""
    X_live = _make_series(30)

    picked = []
    for random_state in range(6):
        detector = _pretrained("pattern", random_state=random_state).fit(X_live)
        assert detector.pattern_ in detector.patterns_
        picked.append(detector.pattern_)

    # the pick is random, so both stored patterns are seen over these seeds
    assert set(picked) == {(5, 15), (10,)}


def test_pattern_predict_replays_the_picked_pattern():
    """Test predict fires exactly at the positions of the picked pattern."""
    X_live = _make_series(30)

    detector = _pretrained("pattern").fit(X_live)

    assert detector.pattern_ == (5, 15)
    assert _alarms(detector, X_live) == [5, 15]


def test_pattern_predict_ignores_the_values_and_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed, and do not depend on its values."""
    detector = _pretrained("pattern").fit(_make_series(100))

    assert _alarms(detector, _make_series(30, start=100, value=7)) == [5, 15]


def test_pattern_predict_drops_alarms_after_the_end():
    """Test alarms beyond the end of the series passed are not fired."""
    detector = _pretrained("pattern").fit(_make_series(30))

    # the pattern is (5, 15), so only the first alarm fits into 10 points
    assert _alarms(detector, _make_series(10)) == [5]


def test_pattern_fit_without_pretrain_uses_y():
    """Test the pattern is taken from the fit series if there was no pretrain."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [11, 3]})

    detector = DummyPretrainedAnomalies(strategy="pattern").fit(X, y)

    assert detector.pattern_ == (3, 11)
    assert _alarms(detector, _make_series(30)) == [3, 11]


def test_pattern_pretrained_patterns_win_over_fit_y():
    """Test a pretrained pattern is used, even if fit also gets known events."""
    detector = _pretrained("pattern")

    detector.fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3]}))

    assert detector.pattern_ == (5, 15)


def test_pattern_nan_in_ilocs_raises():
    """Test a NaN event position raises a ValueError, in pretrain and in fit."""
    X, _ = _panel_and_events()
    y = _make_events([("a", np.nan), ("b", 10)])

    with pytest.raises(ValueError, match="NaN"):
        DummyPretrainedAnomalies(strategy="pattern").pretrain(X, y)

    with pytest.raises(ValueError, match="NaN"):
        DummyPretrainedAnomalies(strategy="pattern").fit(
            _make_series(20), pd.DataFrame({"ilocs": [3, np.nan]})
        )


# strategy "timefreq"
# -------------------


def test_timefreq_pretrain_counts_events_per_time_from_start():
    """Test pretrain counts events per time-from-start, over the series at risk."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5), ("b", 10)])

    detector = DummyPretrainedAnomalies(strategy="timefreq").pretrain(X, y)

    assert len(detector.event_prob_) == 20
    assert detector.event_counts_[5] == 2
    assert detector.event_counts_[10] == 1
    assert detector.event_counts_.sum() == 3
    assert list(detector.n_at_risk_) == [2] * 20
    assert detector.event_prob_[5] == 1.0
    assert detector.event_prob_[10] == 0.5
    assert detector.event_prob_[0] == 0.0


def test_timefreq_short_series_does_not_shrink_late_probability():
    """Test the denominator is the series at risk, not the total series count."""
    X = _make_panel([20, 5])
    y = _make_events([("a", 10)])

    detector = DummyPretrainedAnomalies(strategy="timefreq").pretrain(X, y)

    # both series reach time-from-start 0, only the long one reaches 10
    assert detector.n_at_risk_[0] == 2
    assert detector.n_at_risk_[10] == 1
    assert detector.event_prob_[10] == 1.0


def test_timefreq_event_past_the_end_of_its_series_is_ignored():
    """Test an event after the end of its own series does not raise the count."""
    X = _make_panel([20, 5])
    y = _make_events([("a", 10), ("b", 10)])

    detector = DummyPretrainedAnomalies(strategy="timefreq").pretrain(X, y)

    # only the 20 point series reaches time-from-start 10, so only its event counts
    assert detector.n_at_risk_[10] == 1
    assert detector.event_counts_[10] == 1
    assert detector.event_prob_[10] == 1.0
    assert detector.event_prob_.max() <= 1.0


def test_timefreq_pretrain_without_y_learns_zero_probabilities():
    """Test pretrain without known events learns zero probabilities."""
    X = _make_panel([20, 20])

    detector = DummyPretrainedAnomalies(strategy="timefreq").pretrain(X)

    assert detector.event_counts_.sum() == 0
    assert not detector.event_prob_.any()


def test_timefreq_pretrain_flattens_y_instance_labels():
    """Test instance labels of y are flattened the same way as those of X."""
    X, y = _hierarchical_panel_and_events([3, 3, 3, 7])

    detector = DummyPretrainedAnomalies(strategy="timefreq").pretrain(X, y)

    # 4 series of 10 points, the event of the instance not in X is ignored
    assert list(detector.n_at_risk_) == [4] * 10
    assert detector.event_counts_[3] == 3
    assert detector.event_counts_[7] == 0
    assert detector.event_prob_[3] == 0.75


def test_timefreq_second_pretrain_replaces():
    """Test a second pretrain replaces the counts of the first."""
    X = _make_panel([20, 20])
    detector = DummyPretrainedAnomalies(strategy="timefreq")
    detector.pretrain(X, _make_events([("a", 5), ("b", 5)]))
    assert detector.event_prob_[5] == 1.0

    detector.pretrain(X, _make_events([("a", 12)]))

    assert detector.state == "pretrained"
    assert detector.event_prob_[5] == 0.0
    assert detector.event_prob_[12] == 0.5


@pytest.mark.parametrize("random_state", [0, 1, 2, 3])
def test_timefreq_probability_one_always_fires_and_zero_never(random_state):
    """Test a time-from-start of probability one fires, and zero does not."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])
    X_live = _make_series(20)

    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=random_state)
    detector.pretrain(X, y).fit(X_live)

    assert _alarms(detector, X_live) == [5]


def test_timefreq_predict_ignores_the_values_and_is_on_the_series_passed():
    """Test alarms are ilocs on the X passed, and do not depend on its values."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])

    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=1)
    detector.pretrain(X, y).fit(_make_series(100))

    assert _alarms(detector, _make_series(20, start=100, value=7)) == [5]


def test_timefreq_predict_fires_nothing_past_the_learnt_horizon():
    """Test no alarm is fired beyond the last time-from-start seen in pretrain."""
    X = _make_panel([5, 5])
    # every time-from-start of both series has an event, so the curve is all ones
    y = _make_events([(name, t) for name in ["a", "b"] for t in range(5)])

    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=0)
    detector.pretrain(X, y).fit(_make_series(12))

    assert _alarms(detector, _make_series(12)) == [0, 1, 2, 3, 4]


def test_timefreq_same_seed_gives_the_same_alarms():
    """Test a fixed random_state fires the same alarms on the same length."""
    X_live = _make_series(20)

    def alarms(random_state):
        detector = _pretrained("timefreq", random_state=random_state)
        return _alarms(detector.fit(X_live), X_live)

    # one of the two series has an event at every time-from-start
    assert (_pretrained("timefreq").event_prob_ == 0.5).all()

    assert alarms(0) == alarms(0)
    assert alarms(0) != alarms(1)


def test_timefreq_same_seed_same_cursor_same_length_same_alarms():
    """Test the same seed, cursor and length of X give the same alarms."""
    X = _make_series(20)
    first = _pretrained("timefreq", 3).fit(X.iloc[:4]).update(X.iloc[4:14])
    second = _pretrained("timefreq", 3).fit(X.iloc[:4]).update(X.iloc[4:14])
    assert first.cursor_ == second.cursor_ == 4

    alarms = _alarms(first, X.iloc[4:14])

    assert alarms == _alarms(second, X.iloc[4:14])
    assert alarms == _alarms(first, X.iloc[4:14])


def test_timefreq_predict_does_not_change_the_detector():
    """Test predict draws without writing anything to the detector."""
    X_live = _make_series(20)

    detector = _pretrained("timefreq").fit(X_live)
    prob_before = detector.prob_.copy()
    keys_before = set(vars(detector))

    detector.predict(X_live)

    assert set(vars(detector)) == keys_before
    assert np.array_equal(detector.prob_, prob_before)


@pytest.mark.parametrize("random_state", [0, 1])
def test_timefreq_fit_without_pretrain_uses_y(random_state):
    """Test probability one at the event offsets of the fit series."""
    X = _make_series(20)
    y = pd.DataFrame({"ilocs": [3, 11]})

    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=random_state)
    detector.fit(X, y)

    assert list(np.flatnonzero(detector.prob_ == 1.0)) == [3, 11]
    assert _alarms(detector, X) == [3, 11]


def test_timefreq_pretrained_probabilities_win_over_fit_y():
    """Test the pretrained curve is used, even if fit also gets known events."""
    X = _make_panel([20, 20])
    y = _make_events([("a", 5), ("b", 5)])

    detector = DummyPretrainedAnomalies(strategy="timefreq", random_state=0)
    detector.pretrain(X, y).fit(_make_series(20), pd.DataFrame({"ilocs": [1, 2, 3]}))

    assert np.array_equal(detector.prob_, detector.event_prob_)
    assert _alarms(detector, _make_series(20)) == [5]
