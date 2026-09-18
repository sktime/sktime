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


@SKIP_IF_UNCHANGED
def test_match_integer_index():
    """Windows are in index steps if the index is integer."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [3, 8]})

    # window is [3, 5], so the alarm at 3 hits and the alarm at 8 does not
    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=2)

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

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=pd.Timedelta("2s"))

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [0]
    assert list(match.false_alarm) == [False, True]

    # times come back on the clock of X, so later metrics get a duration
    advance = match.event_times[0] - match.alarm_times[0]
    assert advance == pd.Timedelta("2s")


@SKIP_IF_UNCHANGED
def test_match_time_index_rejects_step_tolerance():
    """A step count is refused if the index is a time index."""
    X = _make_X(time_index=True)
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [3]})

    with pytest.raises(TypeError, match="max_lead"):
        _match_alarms_to_events(y_true, y_pred, X, max_lead=2)


@SKIP_IF_UNCHANGED
def test_match_no_events():
    """With no true events, every alarm is a false alarm."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": []})
    y_pred = pd.DataFrame({"ilocs": [2, 6]})

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=2)

    assert len(match.hit) == 0
    assert len(match.earliest_hit) == 0
    assert list(match.false_alarm) == [True, True]


@SKIP_IF_UNCHANGED
def test_match_no_alarms():
    """With no alarms, no event is hit, and there is no false alarm."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [4, 7]})
    y_pred = pd.DataFrame({"ilocs": []})

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=2)

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

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=2, max_delay=1)

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [1]
    assert list(match.false_alarm) == [True, False, False, True]


@SKIP_IF_UNCHANGED
def test_match_default_max_delay_is_zero():
    """By default, an alarm after the event does not count as a hit."""
    X = _make_X()
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [6]})

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=3)

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

    match = _match_alarms_to_events(y_true, y_pred, X, max_lead=3)

    assert list(match.hit) == [True]
    assert list(match.earliest_hit) == [1]
    assert list(match.false_alarm) == [False, False, True]
