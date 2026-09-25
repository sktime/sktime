"""Tests for DetectionBenchmark: pretrain isolation, live replay, and scoring."""

__author__ = ["yash-sangwan"]

import numpy as np
import pandas as pd
import pytest

from sktime.benchmarking.detection import (
    DetectionBenchmark,
    _events_after_warmup,
    _leave_one_series_out,
    _replay_live,
)
from sktime.detection.base import BaseDetector
from sktime.detection.dummy import (
    DummyPatternAnomalies,
    DummyRateAnomalies,
    DummyTimeFreqAnomalies,
)
from sktime.performance_metrics.detection import (
    EventTPR,
    FalseAlarmRate,
    MeanDetectionOffset,
)
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed(["sktime.benchmarking", "sktime.detection"]),
    reason="module not changed",
)

LENGTHS = [10, 20, 15]
NAMES = ["a", "b", "c"]
EVENT_ILOCS = [3, 7, 5]


def _make_panel():
    """Make a toy panel of 3 series, of 10, 20 and 15 time points."""
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name, n in zip(NAMES, LENGTHS) for t in range(n)],
        names=["instance", "time"],
    )
    return pd.DataFrame({"value": np.arange(float(len(index)))}, index=index)


def _make_events():
    """Make toy known events, one per series, as offsets into their own series."""
    return pd.DataFrame(
        {"ilocs": EVENT_ILOCS},
        index=pd.MultiIndex.from_tuples(
            [(name, 0) for name in NAMES], names=["instance", "event_no"]
        ),
    )


def _make_series(n_timepoints):
    """Make a single toy series of n_timepoints points."""
    return pd.DataFrame(
        {"value": np.arange(float(n_timepoints))},
        index=pd.RangeIndex(n_timepoints),
    )


class _ChunkPositionDetector(BaseDetector):
    """Test detector firing at fixed positions inside every series passed.

    Records the known events seen in ``fit`` and ``update`` in ``y_seen_``,
    so a test can check that the replay passes none.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": False,
        "tests:skip_all": True,
    }

    def __init__(self, positions=(0,)):
        self.positions = positions
        super().__init__()

    def _fit(self, X, y=None):
        self.y_seen_ = [y]
        return self

    def _update(self, X, y=None):
        self.y_seen_.append(y)
        return self

    def _predict(self, X):
        ilocs = [position for position in self.positions if position < len(X)]
        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()
        return pd.Series(ilocs, dtype="int64")


def test_live_series_is_not_in_the_pretrain_panel():
    """Test the live series of a fold is held out of the pretrain panel."""
    X, y = _make_panel(), _make_events()

    folds = list(_leave_one_series_out(X, y))

    assert len(folds) == len(NAMES)
    for instance, X_pretrain, y_pretrain, X_live, y_live in folds:
        pretrain_instances = set(X_pretrain.index.droplevel(-1))
        assert instance not in pretrain_instances
        assert pretrain_instances == set(NAMES) - {instance}

        # the known events of the live series are held out as well
        assert instance not in set(y_pretrain.index.droplevel(-1))
        assert len(y_pretrain) == len(NAMES) - 1
        assert len(y_live) == 1

        # pretrain panel and live series partition the panel, without overlap
        assert len(X_pretrain) + len(X_live) == len(X)
        assert len(X_live) == LENGTHS[NAMES.index(instance)]
        # the live series is a single series, ready for fit and predict
        assert isinstance(X_live, pd.DataFrame)
        assert not isinstance(X_live.index, pd.MultiIndex)


def test_pretrain_stats_match_the_pretrain_panel():
    """Test a detector pretrained by a fold sees the other series only."""
    X, y = _make_panel(), _make_events()

    for instance, X_pretrain, y_pretrain, X_live, _ in _leave_one_series_out(X, y):
        detector = DummyRateAnomalies().pretrain(X_pretrain, y_pretrain)

        assert detector.n_pretrain_timepoints_ == len(X) - len(X_live)
        assert detector.n_pretrain_timepoints_ == len(X_pretrain)
        assert detector.n_pretrain_events_ == len(NAMES) - 1

        other = DummyTimeFreqAnomalies().pretrain(X_pretrain, y_pretrain)

        # the curve is as long as the longest series that was pretrained on
        other_lengths = [n for name, n in zip(NAMES, LENGTHS) if name != instance]
        assert len(other.event_prob_) == max(other_lengths)


def test_run_validation_pretrains_one_fold_per_series():
    """Test the benchmark runs one fold per series of the panel."""
    benchmark = DetectionBenchmark()
    benchmark.add_task((_make_panel(), _make_events()), task_id="toy")
    task = benchmark.tasks.entities["toy"]

    folds = benchmark._run_validation(task, DummyRateAnomalies())

    assert list(folds) == list(range(len(NAMES)))
    # the task has no scorers, so no fold has scores
    assert all(fold.scores == {} for fold in folds.values())


def test_pretrained_detector_raises():
    """Test a detector that is not in state new when pretrained is refused."""
    X, y = _make_panel(), _make_events()

    benchmark = DetectionBenchmark()
    benchmark.add_task((X, y), task_id="toy")
    task = benchmark.tasks.entities["toy"]

    # a detector that already carries pretrained state, as its clone would
    pretrained = DummyRateAnomalies().pretrain(X, y)
    assert pretrained.clone().state == "pretrained"

    with pytest.raises(ValueError, match="must be in state 'new'"):
        benchmark._run_validation(task, pretrained)


def test_warmup_alarms_are_dropped():
    """Test the warm-up prefix is not replayed, so it raises no alarm."""
    detector = _ChunkPositionDetector(positions=(0,))

    alarms = _replay_live(detector, _make_series(10), warmup=3, chunk_size=1)

    # the detector fires on every chunk, but the first 3 points are the fit data
    assert list(alarms["ilocs"]) == [3, 4, 5, 6, 7, 8, 9]
    assert alarms["ilocs"].min() >= 3


def test_chunk_size_one_keeps_raw_ilocs():
    """Test a chunk of one point credits the alarm at its own position."""
    detector = _ChunkPositionDetector(positions=(0,))

    alarms = _replay_live(detector, _make_series(6), warmup=1, chunk_size=1)

    # end - 1 == start for a chunk of one point, so the mapping is the identity
    assert list(alarms["ilocs"]) == [1, 2, 3, 4, 5]


def test_mid_chunk_alarm_is_credited_at_the_chunk_end():
    """Test an alarm inside a chunk is reported at the last point of the chunk."""
    detector = _ChunkPositionDetector(positions=(1,))

    # chunks are [2, 6) and [6, 10), so the alarms land at 5 and 9
    alarms = _replay_live(detector, _make_series(10), warmup=2, chunk_size=4)

    assert list(alarms["ilocs"]) == [5, 9]
    # not at the raw position inside the chunk, which would be 3 and 7
    assert 3 not in list(alarms["ilocs"])
    assert 7 not in list(alarms["ilocs"])

    # several alarms in one chunk collapse into the one at the chunk end
    many = _ChunkPositionDetector(positions=(0, 1, 2))
    assert list(_replay_live(many, _make_series(10), 2, 4)["ilocs"]) == [5, 9]


def test_no_known_events_are_passed_to_update():
    """Test the replay passes no y to fit, update, or update_predict."""
    detector = _ChunkPositionDetector()

    _replay_live(detector, _make_series(6), warmup=2, chunk_size=2)

    # one entry from fit, one per update_predict chunk
    assert len(detector.y_seen_) == 3
    assert all(y_seen is None for y_seen in detector.y_seen_)
    assert detector._y is None


def test_run_validation_replays_every_live_series():
    """Test the benchmark replays the live series of every fold."""
    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_task(
        (_make_panel(), _make_events()), task_id="toy", warmup=2, chunk_size=1
    )
    task = benchmark.tasks.entities["toy"]

    folds = benchmark._run_validation(task, _ChunkPositionDetector(positions=(0,)))

    assert list(folds) == list(range(len(NAMES)))
    for i in range(len(NAMES)):
        alarms = folds[i].predictions
        # the detector fires on every chunk after the warm-up
        assert list(alarms["ilocs"]) == list(range(2, LENGTHS[i]))
        # the known events of the live series come back as the ground truth
        assert len(folds[i].ground_truth) == 1
        assert folds[i].ground_truth["ilocs"].iloc[0] == EVENT_ILOCS[i]


def _make_scorers():
    """Make the three live detection metrics, with explicit units.

    The toy series have a RangeIndex, so offsets are in index units and
    ``time_unit`` is ignored. It is passed explicitly all the same, so the
    test does not depend on the default of a metric.
    """
    return [
        EventTPR(min_offset=-1, max_offset=0),
        MeanDetectionOffset(min_offset=-1, max_offset=0, time_unit="s"),
        FalseAlarmRate(min_offset=-1, max_offset=0, time_unit="s"),
    ]


def _make_scoring_panel():
    """Make a panel of 2 series of 10 points, each with events at 4 and 9."""
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name in ["a", "b"] for t in range(10)],
        names=["instance", "time"],
    )
    X = pd.DataFrame({"value": np.arange(float(len(index)))}, index=index)
    y = pd.DataFrame(
        {"ilocs": [4, 9, 4, 9]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
            names=["instance", "event_no"],
        ),
    )
    return X, y


def test_result_table_has_the_three_score_columns():
    """Test the three metrics reach the results table of a run."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(_ChunkPositionDetector(positions=(1,)))
    benchmark.add_task(
        _make_scoring_panel(), _make_scorers(), task_id="toy", warmup=2, chunk_size=4
    )

    results = benchmark.run()

    assert len(results) == 1
    for name in ["EventTPR", "MeanDetectionOffset", "FalseAlarmRate"]:
        columns = [column for column in results.columns if column.startswith(name)]
        # one column per fold, plus the mean and the standard deviation
        assert f"{name}_fold_0_test" in columns
        assert f"{name}_mean" in columns


def test_event_tpr_on_a_toy_series():
    """Test one hand-checkable EventTPR score of a fold."""
    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_task(
        _make_scoring_panel(), _make_scorers(), task_id="toy", warmup=2, chunk_size=4
    )
    task = benchmark.tasks.entities["toy"]

    folds = benchmark._run_validation(task, _ChunkPositionDetector(positions=(1,)))

    # chunks are [2, 6) and [6, 10), so the alarms are credited at 5 and 9
    assert list(folds[0].predictions["ilocs"]) == [5, 9]
    # events are at 4 and 9, the hit window is [T - 1, T]
    # event 4: window [3, 4], no alarm, missed
    # event 9: window [8, 9], alarm at 9, hit
    assert folds[0].scores["EventTPR"] == 0.5
    assert folds[1].scores["EventTPR"] == 0.5


def test_empty_alarms_still_produce_a_row():
    """Test a detector that never fires is still scored, and still reported."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(_ChunkPositionDetector(positions=()), "NeverFires")
    benchmark.add_task(
        _make_scoring_panel(), _make_scorers(), task_id="toy", warmup=2, chunk_size=4
    )

    results = benchmark.run()

    assert len(results) == 1
    assert results["model_id"].iloc[0] == "NeverFires"
    # no alarm hits any event, and no alarm is a false alarm
    assert results["EventTPR_fold_0_test"].iloc[0] == 0.0
    assert results["FalseAlarmRate_fold_0_test"].iloc[0] == 0.0


def test_warmup_events_are_not_scored():
    """Test known events inside the warm-up are dropped before scoring."""
    X, y = _make_scoring_panel()
    y_live = y.loc["a"]

    kept = _events_after_warmup(y_live, warmup=5)

    assert list(kept["ilocs"]) == [9]
    assert list(_events_after_warmup(y_live, warmup=2)["ilocs"]) == [4, 9]
    assert _events_after_warmup(None, warmup=2) is None


def test_scorers_without_known_events_raise():
    """Test scoring a task that carries no known events is refused."""
    X, _ = _make_scoring_panel()

    benchmark = DetectionBenchmark()
    benchmark.add_task(X, _make_scorers(), task_id="no_events")
    task = benchmark.tasks.entities["no_events"]

    with pytest.raises(ValueError, match="registered without"):
        benchmark._run_validation(task, _ChunkPositionDetector())


def test_live_series_shorter_than_the_warmup():
    """Test a live series shorter than the warm-up is fitted, and not replayed."""
    detector = _ChunkPositionDetector(positions=(0,))

    alarms = _replay_live(detector, _make_series(2), warmup=5, chunk_size=1)

    assert detector.is_fitted
    assert len(alarms) == 0
    assert list(alarms.columns) == ["ilocs"]
    # there is nothing left to replay, so update is never called
    assert len(detector.y_seen_) == 1


def test_estimator_that_is_not_a_detector_raises():
    """Test only detectors can be added to the benchmark."""
    from sktime.forecasting.naive import NaiveForecaster

    benchmark = DetectionBenchmark()

    with pytest.raises(TypeError, match="benchmarks detectors"):
        benchmark.add_estimator(NaiveForecaster())


def _panel_with_lengths(lengths):
    """Make a panel with one series per entry of lengths, named a, b, c, ..."""
    names = [chr(ord("a") + i) for i in range(len(lengths))]
    index = pd.MultiIndex.from_tuples(
        [(name, t) for name, n in zip(names, lengths) for t in range(n)],
        names=["instance", "time"],
    )
    return pd.DataFrame({"value": np.arange(float(len(index)))}, index=index)


def _events_at(pairs):
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


def test_run_validation_pretrains_on_the_held_out_panel():
    """Test a fold pretrains on the other series, not on the live series too."""
    # only c reaches time-from-start 5, and only c has an event there, so a
    # detector that saw c would learn probability one at 5 and fire on it
    X = _panel_with_lengths([3, 3, 8])
    y = _events_at([("c", 5)])

    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_task((X, y), task_id="held_out", warmup=1, chunk_size=7)
    task = benchmark.tasks.entities["held_out"]

    folds = benchmark._run_validation(task, DummyTimeFreqAnomalies(random_state=0))

    # the fold of c pretrains on a and b, which carry no events at all
    assert len(folds[2].predictions) == 0

    # control: the same replay does fire when the event is in the pretrain panel
    X_control = _panel_with_lengths([8, 8, 8])
    y_control = _events_at([("a", 5), ("b", 5)])

    control = DetectionBenchmark(return_data=True)
    control.add_task((X_control, y_control), task_id="control", warmup=1, chunk_size=7)
    control_task = control.tasks.entities["control"]

    control_folds = control._run_validation(
        control_task, DummyTimeFreqAnomalies(random_state=0)
    )

    assert list(control_folds[2].predictions["ilocs"]) == [7]


def test_run_validation_drops_warmup_events_before_scoring():
    """Test a known event inside the warm-up is not scored against the alarms."""
    X = _panel_with_lengths([10, 10])
    y = _events_at([("a", 1), ("a", 9), ("b", 1), ("b", 9)])

    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_task(
        (X, y),
        [EventTPR(min_offset=-1, max_offset=0)],
        task_id="warmup",
        warmup=4,
        chunk_size=4,
    )
    task = benchmark.tasks.entities["warmup"]

    folds = benchmark._run_validation(task, _ChunkPositionDetector(positions=(1,)))

    # chunks are [4, 8) and [8, 10), so the alarms are credited at 7 and 9
    assert list(folds[0].predictions["ilocs"]) == [7, 9]
    # the event at 1 is inside the warm-up, so it is dropped before scoring
    assert list(folds[0].ground_truth["ilocs"]) == [9]
    # event 9 has the hit window [8, 9] and the alarm at 9 hits it
    # keeping the event at 1, which no alarm could reach, would give 0.5
    assert folds[0].scores["EventTPR"] == 1.0
    assert folds[1].scores["EventTPR"] == 1.0


class _SeenSeriesDetector(BaseDetector):
    """Test detector that fires on a series it saw in pretrain.

    Remembers the values of every series it pretrains on. Its
    ``_pretrain_update`` adds to what it saw before, instead of replacing it,
    as a detector that keeps learning would. Reused across folds, it would
    carry the pretrain panel of an earlier fold, which holds the live series
    of a later fold, and fire on that live series.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:pretrain": True,
        "fit_is_empty": False,
        "tests:skip_all": True,
    }

    def _pretrain(self, X, y=None):
        self.seen_values_ = set(X.iloc[:, 0])
        return self

    def _pretrain_update(self, X, y=None):
        # accumulates, a second pretrain keeps what the first one saw
        self.seen_values_ = self.seen_values_ | set(X.iloc[:, 0])
        return self

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        if X.iloc[0, 0] in getattr(self, "seen_values_", set()):
            return pd.Series([0], dtype="int64")
        return BaseDetector._empty_sparse()


def test_reusing_one_detector_across_folds_would_leak():
    """Test no fold pretrains on its own live series, even if pretrain accumulates.

    Every value of the panel belongs to one series only, so the detector
    fires exactly when it has pretrained on the live series. This is the
    cross-fold leak of global forecasting evaluate: a model carried from one
    fold to the next keeps the pretrain panel of the earlier fold.
    """
    X = _panel_with_lengths([5, 5, 5])

    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_task(X, task_id="reuse", warmup=1, chunk_size=1)
    task = benchmark.tasks.entities["reuse"]

    detector = _SeenSeriesDetector()
    folds = benchmark._run_validation(task, detector)

    # each fold pretrains a fresh clone, which never saw its live series
    assert all(len(fold.predictions) == 0 for fold in folds.values())
    # the detector passed in is neither pretrained nor fitted by the benchmark
    assert detector.state == "new"

    # control: one detector reused for the first two folds does leak
    (_, X_pretrain_0, _, X_live_0, _), (_, X_pretrain_1, _, X_live_1, _), _ = list(
        _leave_one_series_out(X)
    )
    reused = _SeenSeriesDetector()
    reused.pretrain(X_pretrain_0)
    _replay_live(reused, X_live_0, warmup=1, chunk_size=1)

    # fitted, so this pretrain runs _pretrain_update and keeps the first panel,
    # which holds the live series of the second fold
    reused.pretrain(X_pretrain_1)

    assert len(_replay_live(reused, X_live_1, warmup=1, chunk_size=1)) > 0


def test_pattern_dummy_keeps_time_from_start_across_chunks():
    """Test DummyPatternAnomalies hits every event with chunks of one point."""
    names = ["a", "b", "c"]
    X = _panel_with_lengths([30, 30, 30])
    y = _events_at([(name, t) for name in names for t in (9, 19, 29)])

    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_estimator(DummyPatternAnomalies(random_state=0))
    benchmark.add_task(
        (X, y),
        [EventTPR(min_offset=0, max_offset=0)],
        task_id="stream",
        warmup=1,
        chunk_size=1,
    )

    results = benchmark.run()

    # every series has events at 9, 19 and 29, so every stored pattern is
    # (9, 19, 29), and a replay that keeps time-from-start hits all of them
    for fold in range(len(names)):
        assert results[f"EventTPR_fold_{fold}_test"].iloc[0] == 1.0
        # EventTPR ignores false alarms, so pin the alarms themselves:
        # exactly the events, and nothing else
        alarms = results[f"predictions_fold_{fold}"].iloc[0]
        assert list(alarms["ilocs"]) == [9, 19, 29]
    assert results["EventTPR_mean"].iloc[0] == 1.0


def test_chunk_size_is_set_per_task():
    """Test two tasks on the same panel, chunk sizes 1 and 4, credit alarms apart."""
    names = ["a", "b", "c"]
    X = _panel_with_lengths([30, 30, 30])
    y = _events_at([(name, t) for name in names for t in (9, 19, 29)])

    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_estimator(DummyPatternAnomalies(random_state=0))
    benchmark.add_task((X, y), warmup=1, chunk_size=1)
    benchmark.add_task((X, y), warmup=1, chunk_size=4)

    results = benchmark.run().set_index("validation_id")

    # the default ids differ by chunk size, so the two tasks do not collide
    id_1 = "[dataset=_]_[split=leave_one_series_out]_[warmup=1]_[chunk_size=1]"
    id_4 = "[dataset=_]_[split=leave_one_series_out]_[warmup=1]_[chunk_size=4]"
    assert sorted(results.index) == sorted([id_1, id_4])

    # chunks of one point report every event where it happens
    alarms_1 = results.loc[id_1, "predictions_fold_0"]
    assert list(alarms_1["ilocs"]) == [9, 19, 29]

    # chunks of four are [1, 5), [5, 9), [9, 13), ..., [25, 29), [29, 30), and
    # each event is credited at the end of its chunk: 9 at 12, 19 at 20, 29 at 29
    alarms_4 = results.loc[id_4, "predictions_fold_0"]
    assert list(alarms_4["ilocs"]) == [12, 20, 29]


@pytest.mark.parametrize(
    "setting",
    [{"warmup": 0}, {"chunk_size": 0}, {"warmup": 1.5}],
    ids=["warmup_0", "chunk_size_0", "warmup_not_int"],
)
def test_invalid_replay_settings_raise_in_add_task(setting):
    """Test add_task refuses a warm-up or chunk size below 1, or not an integer."""
    benchmark = DetectionBenchmark()

    with pytest.raises(ValueError, match="must be an integer of at least 1"):
        benchmark.add_task((_make_panel(), _make_events()), **setting)

    # the task is refused before anything is registered
    assert benchmark.tasks.entities == {}


@pytest.mark.parametrize(
    "setting", [{"warmup": 2}, {"chunk_size": 2}], ids=["warmup", "chunk_size"]
)
def test_replay_settings_cannot_be_set_on_the_constructor(setting):
    """Test the constructor refuses warmup and chunk_size, which go to add_task."""
    with pytest.raises(TypeError):
        DetectionBenchmark(**setting)
