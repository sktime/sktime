"""Tests for matching detected alarms to true events."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection.utils._match import _match_alarms_to_events
from sktime.tests.test_switch import run_test_module_changed

SKIP_IF_UNCHANGED = pytest.mark.skipif(
    not run_test_module_changed("sktime.performance_metrics.detection"),
    reason="run test only if detection module changed",
)


def _make_X(n_timepoints=10, time_index=False):
    """Make a series to map event positions onto."""
    if time_index:
        index = pd.date_range("2020-01-01", periods=n_timepoints, freq="s")
        return pd.DataFrame({"foo": range(n_timepoints)}, index=index)
    return pd.DataFrame({"foo": range(n_timepoints)})


def _make_uneven_time_X():
    """Make a series on an uneven clock: steps of 10s, then one step of 4s."""
    seconds = [0, 10, 20, 30, 34]
    index = pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s")
    return pd.DataFrame({"foo": range(len(seconds))}, index=index)


@SKIP_IF_UNCHANGED
def test_match_integer_index():
    """Windows are in the units of an integer index."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [3, 8]})

    # window is [3, 5], so the alarm at 3 hits and the alarm at 8 does not
    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [0]
    assert list(match.false_alarm) == [False, True]
    assert match.event_times[0] == 5
    assert match.alarm_times[0] == 3


@SKIP_IF_UNCHANGED
def test_match_time_index():
    """Windows are in time units if the index is a time index."""
    X = _make_X(time_index=True)  # one second per step
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [3, 8]})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=pd.Timedelta("-2s"))

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [0]
    assert list(match.false_alarm) == [False, True]

    # times come back on the clock of X, so later metrics get a duration
    advance = match.event_times[0] - match.alarm_times[0]
    assert advance == pd.Timedelta("2s")


@SKIP_IF_UNCHANGED
def test_match_time_index_rejects_step_offset():
    """A plain number is refused as an offset if the index is a time index."""
    X = _make_X(time_index=True)
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [3]})

    with pytest.raises(TypeError, match="min_offset"):
        _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)


@SKIP_IF_UNCHANGED
def test_match_no_events():
    """With no true events, every alarm is a false alarm."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": []})
    y_pred = pd.DataFrame({"ilocs": [2, 6]})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)

    assert len(match.hit) == 0
    assert len(match.earliest_hit) == 0
    assert list(match.false_alarm) == [True, True]


@SKIP_IF_UNCHANGED
def test_match_no_alarms():
    """With no alarms, no event is hit, and there is no false alarm."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [4, 7]})
    y_pred = pd.DataFrame({"ilocs": []})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)

    assert list(match.hit) == [False, False]
    assert list(match.earliest_hit) == [-1, -1]
    assert len(match.false_alarm) == 0


@SKIP_IF_UNCHANGED
def test_match_window_edges():
    """Alarms on both window edges are hits, one step outside is not."""
    X = _make_X(n_timepoints=20)
    y_true = pd.DataFrame({"ilocs": [10]})
    # window is [8, 11], so 8 and 11 are hits, 7 and 12 are not
    y_pred = pd.DataFrame({"ilocs": [7, 8, 11, 12]})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-2, max_offset=1)

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [1]
    assert list(match.false_alarm) == [True, False, False, True]


# event at 10; the first alarm is always just outside the early edge,
# so the earliest hit is always the alarm at position 1
WINDOW_CASES = [
    pytest.param(0, 0, [9, 10, 11], [True, False, True], id="exact_only"),
    pytest.param(-3, 0, [6, 7, 10, 11], [True, False, False, True], id="advance_only"),
    pytest.param(0, 2, [9, 10, 12, 13], [True, False, False, True], id="late_only"),
    pytest.param(
        -3, 2, [6, 7, 12, 13], [True, False, False, True], id="before_and_after"
    ),
    pytest.param(
        -5, -2, [4, 5, 8, 9, 10], [True, False, False, True, True], id="both_negative"
    ),
]


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize("min_offset, max_offset, alarms, false_alarm", WINDOW_CASES)
def test_match_window_cases(min_offset, max_offset, alarms, false_alarm):
    """The window is [T + min_offset, T + max_offset], with signed offsets.

    In the both-negative case, the window is [5, 8] for the event at 10, so
    alarms that are not early enough, including one at the event, do not count.
    """
    X = _make_X(n_timepoints=20)
    y_true = pd.DataFrame({"ilocs": [10]})
    y_pred = pd.DataFrame({"ilocs": alarms})

    match = _match_alarms_to_events(
        y_true,
        y_pred,
        X,
        min_offset=min_offset,
        max_offset=max_offset,
    )

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [1]
    assert list(match.false_alarm) == false_alarm


@SKIP_IF_UNCHANGED
def test_match_default_max_offset_is_zero():
    """By default, an alarm after the event does not count as a hit."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [6]})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-3)

    assert list(match.hit) == [False]
    assert list(match.earliest_hit) == [-1]
    assert list(match.false_alarm) == [True]


@SKIP_IF_UNCHANGED
def test_match_extra_alarms_in_one_window():
    """Extra alarms in one window are not false alarms, the earliest is kept."""
    X = _make_X(n_timepoints=20)
    y_true = pd.DataFrame({"ilocs": [10]})
    # window is [7, 10], alarms are out of order on purpose
    y_pred = pd.DataFrame({"ilocs": [9, 8, 15]})

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-3)

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [1]
    assert list(match.false_alarm) == [False, False, True]


@SKIP_IF_UNCHANGED
def test_match_uneven_time_index():
    """Windows are measured in time, not in steps.

    Both alarms are one step before their event. The first is 10s early and
    misses the 5s window, the second is 4s early and hits it. Counting steps
    would treat both alarms the same, so it would fail this test.
    """
    X = _make_uneven_time_X()
    y_true = pd.DataFrame({"ilocs": [2, 4]})  # events at 20s and 34s
    y_pred = pd.DataFrame({"ilocs": [1, 3]})  # alarms at 10s and 30s

    match = _match_alarms_to_events(y_true, y_pred, X, min_offset=pd.Timedelta("-5s"))

    assert list(match.hit) == [False, True]
    assert list(match.earliest_hit) == [-1, 1]
    assert list(match.false_alarm) == [True, False]


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize("bad_iloc", [-1, 10])
@pytest.mark.parametrize("bad_table", ["y_true", "y_pred"])
def test_match_rejects_ilocs_outside_X(bad_iloc, bad_table):
    """Positions outside [0, len(X)) raise, and -1 does not wrap to the end."""
    X = _make_X()  # 10 points, so valid positions are 0 to 9
    good = pd.DataFrame({"ilocs": [5]})
    bad = pd.DataFrame({"ilocs": [bad_iloc]})
    y_true, y_pred = (bad, good) if bad_table == "y_true" else (good, bad)

    with pytest.raises(ValueError, match=f"{bad_table} has 'ilocs' outside"):
        _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize("time_index", [False, True])
def test_match_rejects_min_after_max(time_index):
    """An inverted window raises, instead of silently matching nothing."""
    X = _make_X(time_index=time_index)
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [4]})

    step = pd.Timedelta("1s") if time_index else 1

    with pytest.raises(ValueError, match="min_offset must not be after"):
        _match_alarms_to_events(
            y_true, y_pred, X, min_offset=1 * step, max_offset=-1 * step
        )


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize("name", ["min_offset", "max_offset"])
def test_match_rejects_nan_offset(name):
    """A NaN offset raises, instead of silently bending the window."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [4]})

    offsets = {"min_offset": -2, "max_offset": 0}
    offsets[name] = float("nan")

    with pytest.raises(ValueError, match=f"{name} must not be NaN"):
        _match_alarms_to_events(y_true, y_pred, X, **offsets)
