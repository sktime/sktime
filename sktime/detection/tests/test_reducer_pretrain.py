"""Tests for ReducerPretrainDetector, detection by classification of windows."""

__author__ = ["yash-sangwan"]

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from sktime.detection.reduce import ReducerPretrainDetector
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)


def _make_panel(series):
    """Make a panel with one univariate series per array, named a, b, c, ..."""
    names = [chr(ord("a") + i) for i in range(len(series))]
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name, values in zip(names, series) for t in range(len(values))],
        names=["instance", "time"],
    )
    return pd.DataFrame({"value": np.concatenate(series).astype(float)}, index=index)


def _make_events(pairs):
    """Make y in the pretrain format, from (instance, time-from-start) pairs."""
    event_no = {}
    index = []
    for instance, _ in pairs:
        index.append((instance, event_no.get(instance, 0)))
        event_no[instance] = event_no.get(instance, 0) + 1
    return pd.DataFrame(
        {"ilocs": [iloc for _, iloc in pairs]},
        index=pd.MultiIndex.from_tuples(index, names=["instance", "event_no"]),
    )


def _make_series(values):
    """Make a single univariate series from an array."""
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _spike_series(n_timepoints, spikes):
    """Make zeros with a value of 1 at each position in spikes."""
    values = np.zeros(n_timepoints)
    values[list(spikes)] = 1.0
    return values


def _spike_panel_and_events():
    """Make 3 series of 30 points, each with a spike 2 points before its event."""
    events = [10, 15, 20]
    X = _make_panel([_spike_series(30, [event - 2]) for event in events])
    y = _make_events([(name, event) for name, event in zip("abc", events)])
    return X, y


def _spike_detector(**params):
    """Make a detector that learns the spike pattern of the toy panel exactly.

    With detection_offset 1, a positive window ends 1 point before the event,
    so its spike is 1 point before its end, and the detector fires 1 point
    after a spike.
    """
    defaults = {
        "estimator": DecisionTreeClassifier(random_state=0),
        "window_length": 2,
        "detection_offset": 1,
        "negative_window_fraction": 1.0,
    }
    return ReducerPretrainDetector(**{**defaults, **params})


def _pretrained(**params):
    """Make a spike detector pretrained on the toy panel."""
    return _spike_detector(**params).pretrain(*_spike_panel_and_events())


def _alarms(detector, X):
    """Return the alarms of predict on X, as a list of int."""
    return list(detector.predict(X)["ilocs"])


def _replay(detector, X, warmup, chunk_size):
    """Fit on the first warmup points, then update_predict the rest in chunks.

    Returns the alarms as positions in X, and checks that every iloc returned
    by update_predict is inside its chunk.
    """
    detector.fit(X.iloc[:warmup])
    times = []
    for start in range(warmup, len(X), chunk_size):
        chunk = X.iloc[start : start + chunk_size]
        ilocs = list(detector.update_predict(chunk)["ilocs"])
        assert all(0 <= iloc < len(chunk) for iloc in ilocs)
        times += [start + iloc for iloc in ilocs]
    return times


# windows of pretrain
# -------------------


def test_positive_windows_end_detection_offset_before_each_event():
    """Test a positive window ends detection_offset points before its event."""
    detector = ReducerPretrainDetector(window_length=3, detection_offset=2)
    # the values are the positions, so a row shows which points it holds
    values = np.arange(20.0).reshape(-1, 1)

    rows, labels = detector._training_rows(values, np.array([10, 15]))

    # events at 10 and 15, so the windows end at 8 and 13
    assert rows[labels == 1].tolist() == [[6, 7, 8], [11, 12, 13]]


def test_events_without_a_full_window_or_outside_the_series_are_dropped():
    """Test an event too early for a window, or outside its series, is dropped."""
    detector = ReducerPretrainDetector(window_length=3, detection_offset=2)
    values = np.arange(20.0).reshape(-1, 1)

    rows, labels = detector._training_rows(values, np.array([3, -1, 25, 10]))

    # the window of 3 would end at 1, before the first full window at 2,
    # and -1 and 25 are outside the series of 20 points
    assert rows[labels == 1].tolist() == [[6, 7, 8]]


def test_negative_windows_keep_clear_of_events():
    """Test a negative window has no event in it, nor within the offset after it."""
    detector = ReducerPretrainDetector(
        window_length=3, detection_offset=2, negative_window_fraction=1.0
    )
    values = np.arange(40.0).reshape(-1, 1)
    events = np.array([10, 25])

    rows, labels = detector._training_rows(values, events)
    negative_ends = rows[labels == 0][:, -1].astype(int)

    # a window ending at t holds t - 2 to t, and is followed by t + 1 and t + 2
    near_event = {t for event in events for t in range(event - 2, event + 3)}
    assert set(negative_ends) == set(range(2, 40)) - near_event


def test_negative_window_fraction_takes_equally_spaced_windows():
    """Test the fraction sets the number of negative rows, equally spaced."""
    values = np.arange(40.0).reshape(-1, 1)
    no_events = np.empty(0, dtype="int64")

    detector = ReducerPretrainDetector(window_length=3, negative_window_fraction=0.25)
    rows, labels = detector._training_rows(values, no_events)
    ends = rows[:, -1].astype(int)

    # 38 windows end at 2 to 39, and a quarter of them, rounded up, is 10
    assert (labels == 0).all()
    assert len(rows) == 10
    assert ends[0] == 2
    assert ends[-1] == 39
    gaps = np.diff(ends)
    assert gaps.max() - gaps.min() <= 1


def test_multivariate_rows_are_in_time_order():
    """Test a row holds all variables of a point, point after point."""
    detector = ReducerPretrainDetector(window_length=2)
    values = np.column_stack([np.arange(10.0), 100 + np.arange(10.0)])

    rows, labels = detector._training_rows(values, np.array([5]))

    assert rows.shape[1] == 4
    assert rows[labels == 1].tolist() == [[4, 104, 5, 105]]


# pretrain, fit and predict
# -------------------------


def test_pretrain_learns_the_pattern_and_fires_before_the_event():
    """Test the detector fires where the pretrained pattern shows, and not else."""
    detector = _pretrained()
    assert detector.state == "pretrained"
    assert detector.n_pretrain_positive_ == 3

    X_new = _make_series(_spike_series(40, [5, 25]))
    # a spike announces an event 2 points later, the alarm comes 1 point earlier
    assert _alarms(detector.fit(X_new), X_new) == [6, 26]

    X_flat = _make_series(np.zeros(40))
    assert _alarms(detector.fit(X_flat), X_flat) == []


def test_scores_use_only_the_latest_window():
    """Test the score of a point does not change with the points after it."""
    X, y = _spike_panel_and_events()
    # logistic regression gives scores that move with the values
    detector = ReducerPretrainDetector(
        window_length=3, detection_offset=1, negative_window_fraction=1.0
    ).pretrain(X, y)

    x = _spike_series(30, [5, 20])
    scores = detector.fit(_make_series(x)).transform_scores(_make_series(x))
    scores = scores["scores"].to_numpy()

    # the first 2 points have no full window of 3 yet
    assert np.isnan(scores[:2]).all()
    assert not np.isnan(scores[2:]).any()

    for t in [6, 12, 21]:
        x_changed = x.copy()
        x_changed[t + 1 :] = 7.0
        X_changed = _make_series(x_changed)
        changed = detector.fit(X_changed).transform_scores(X_changed)
        changed = changed["scores"].to_numpy()

        assert np.array_equal(changed[: t + 1], scores[: t + 1], equal_nan=True)
        # the points after t do change their own scores
        assert not np.allclose(changed[t + 1 :], scores[t + 1 :])


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_chunked_update_predict_equals_one_shot_predict(chunk_size):
    """Test update_predict on chunks fires where predict on the whole series does."""
    X = _make_series(_spike_series(40, [5, 17, 30]))
    warmup = 3

    # a window of 4 is longer than some chunks, so it must reach back
    expected = _alarms(_pretrained(window_length=4).fit(X), X)
    times = _replay(_pretrained(window_length=4), X, warmup, chunk_size)

    assert expected == [6, 18, 31]
    assert times == [t for t in expected if t >= warmup]


def test_series_input_gives_the_same_alarms():
    """Test a pd.Series gets the alarms of a DataFrame, also via predict_points."""
    x = _spike_series(40, [5, 25])
    detector = _pretrained()

    from_frame = _alarms(detector.fit(_make_series(x)), _make_series(x))
    # predict_points passes X to _predict without converting it to a DataFrame
    from_series = detector.fit(pd.Series(x)).predict_points(pd.Series(x))

    assert from_frame == [6, 26]
    assert list(from_series["ilocs"]) == from_frame


def test_update_does_not_refit(monkeypatch):
    """Test update keeps the classifier, even when it is passed known events."""
    X = _make_series(_spike_series(20, [5]))
    # fit gets known events too, which a pretrained detector ignores, as
    # BaseDetector.update cannot combine the y of update with no y of fit
    detector = _pretrained().fit(X.iloc[:5], pd.DataFrame({"ilocs": [1]}))
    estimator = detector.estimator_

    def no_refit(*args, **kwargs):
        raise AssertionError("update fitted a classifier")

    monkeypatch.setattr(ReducerPretrainDetector, "_fit_classifier", no_refit)

    detector.update(X.iloc[5:10], y=pd.DataFrame({"ilocs": [2]}))

    assert detector.estimator_ is estimator
    assert detector.pretrain_estimator_ is estimator


def test_predict_does_not_change_the_detector():
    """Test predict and the score methods leave the fitted state unchanged."""
    X = _make_series(_spike_series(20, [5, 12]))
    detector = _pretrained().fit(X.iloc[:5]).update(X.iloc[5:12])
    state = {k: v for k, v in vars(detector).items() if not k.startswith("_")}
    arrays = {k: v.copy() for k, v in state.items() if isinstance(v, np.ndarray)}

    detector.predict(X.iloc[5:12])
    detector.transform_scores(X.iloc[5:12])
    detector.predict_scores(X.iloc[5:12])

    after = {k: v for k, v in vars(detector).items() if not k.startswith("_")}
    assert after.keys() == state.keys()
    assert all(after[k] is v for k, v in state.items())
    assert all(np.array_equal(after[k], v) for k, v in arrays.items())


def test_scores_of_alarms_match_transform_scores():
    """Test predict_scores gives the score of each alarm, above the threshold."""
    X = _make_series(_spike_series(30, [5, 20]))
    detector = _pretrained().fit(X)

    alarms = _alarms(detector, X)
    alarm_scores = detector.predict_scores(X).iloc[:, 0].to_numpy()
    scores = detector.transform_scores(X)["scores"].to_numpy()

    assert alarms == [6, 21]
    assert np.array_equal(alarm_scores, scores[alarms])
    assert (alarm_scores > detector.detection_threshold).all()


def test_clone_keeps_the_pretrained_classifier():
    """Test a clone of a pretrained detector keeps its own copy of the classifier."""
    detector = _pretrained()

    detector_clone = detector.clone()

    assert detector_clone.state == "pretrained"
    assert detector_clone.pretrain_estimator_ is not detector.pretrain_estimator_
    X_new = _make_series(_spike_series(40, [5, 25]))
    assert _alarms(detector_clone.fit(X_new), X_new) == [6, 26]


def test_second_pretrain_replaces():
    """Test a second pretrain replaces the classifier of the first."""
    X, y = _spike_panel_and_events()
    detector = _spike_detector().pretrain(X, y)
    assert detector.n_pretrain_positive_ == 3

    # without known events, the second pretrain learns to fire nowhere
    detector.pretrain(X)

    assert detector.state == "pretrained"
    assert detector.n_pretrain_positive_ == 0
    X_new = _make_series(_spike_series(40, [5, 25]))
    assert _alarms(detector.fit(X_new), X_new) == []


def test_fit_without_pretrain():
    """Test fit learns from its own y if not pretrained, and fires nothing without."""
    X = _make_series(_spike_series(30, [5, 20]))

    assert _alarms(_spike_detector().fit(X), X) == []

    # known events 2 points after each spike, as in the pretrain panel
    detector = _spike_detector().fit(X, pd.DataFrame({"ilocs": [7, 22]}))

    X_new = _make_series(_spike_series(20, [10]))
    assert _alarms(detector, X_new) == [11]


def test_one_class_only():
    """Test pretrain with only negative, or only positive, rows does not fail."""
    X, _ = _spike_panel_and_events()
    X_new = _make_series(_spike_series(40, [5, 25]))

    # no known events: negative rows only, nothing is fired
    negative_only = _spike_detector().pretrain(X)
    assert negative_only.n_pretrain_positive_ == 0
    assert negative_only.n_pretrain_negative_ > 0
    assert _alarms(negative_only.fit(X_new), X_new) == []

    # every window of a short series is near an event: positive rows only,
    # so every point with a full window fires
    X_short = _make_panel([np.zeros(3)])
    y_short = _make_events([("a", 1), ("a", 2)])
    positive_only = ReducerPretrainDetector(window_length=2).pretrain(X_short, y_short)
    assert positive_only.n_pretrain_negative_ == 0
    X_five = _make_series(np.zeros(5))
    assert _alarms(positive_only.fit(X_five), X_five) == [1, 2, 3, 4]


def test_events_of_unknown_instances_are_ignored():
    """Test events of series not in X, or of y without instances, are ignored."""
    X, y = _spike_panel_and_events()
    y_unknown = pd.concat([y, _make_events([("z", 12)])])

    detector = _spike_detector().pretrain(X, y_unknown)
    assert detector.n_pretrain_positive_ == 3

    # a y without an instance level cannot be matched to the series of X
    flat_y = pd.DataFrame({"ilocs": [10, 15]})
    assert _spike_detector().pretrain(X, flat_y).n_pretrain_positive_ == 0


# invalid input
# -------------


@pytest.mark.parametrize(
    "params",
    [
        {"window_length": 0},
        {"window_length": 1.5},
        {"window_length": True},
        {"detection_offset": -1},
        {"detection_threshold": 1.5},
        {"detection_threshold": np.nan},
        {"negative_window_fraction": 0},
        {"negative_window_fraction": 1.5},
    ],
    ids=[
        "window_length_0",
        "window_length_float",
        "window_length_bool",
        "detection_offset_negative",
        "detection_threshold_above_1",
        "detection_threshold_nan",
        "negative_window_fraction_0",
        "negative_window_fraction_above_1",
    ],
)
def test_invalid_parameters_raise_in_pretrain_and_fit(params):
    """Test out-of-range parameters raise in pretrain and fit, not in __init__."""
    X, y = _spike_panel_and_events()
    name = next(iter(params))

    detector = ReducerPretrainDetector(**params)

    with pytest.raises(ValueError, match=name):
        detector.pretrain(X, y)
    with pytest.raises(ValueError, match=name):
        ReducerPretrainDetector(**params).fit(_make_series(np.zeros(20)))


def test_estimator_without_predict_proba_raises():
    """Test a classifier without predict_proba is refused."""
    from sklearn.svm import LinearSVC

    X, y = _spike_panel_and_events()
    detector = ReducerPretrainDetector(estimator=LinearSVC())

    with pytest.raises(TypeError, match="predict_proba"):
        detector.pretrain(X, y)


def test_non_integer_ilocs_raise():
    """Test a missing or non-integer event position raises, in pretrain and fit."""
    X, _ = _spike_panel_and_events()
    y = _make_events([("a", np.nan), ("b", 15)])

    with pytest.raises(ValueError, match="integer positions"):
        _spike_detector().pretrain(X, y)
    with pytest.raises(ValueError, match="integer positions"):
        _spike_detector().fit(
            _make_series(np.zeros(20)), pd.DataFrame({"ilocs": [3.5]})
        )
