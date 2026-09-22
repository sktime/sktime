"""Tests for the event true positive rate."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.detection._event_tpr import EventTPR
from sktime.tests.test_switch import run_test_for_class

SKIP_IF_UNCHANGED = pytest.mark.skipif(
    not run_test_for_class(EventTPR),
    reason="run test only if softdeps are present and incrementally (if requested)",
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


def _events(ilocs):
    """Make an event table, typed like detector output, also when empty."""
    return pd.DataFrame({"ilocs": ilocs}, dtype="int64")


@SKIP_IF_UNCHANGED
def test_event_tpr_hit():
    """An alarm inside the window hits the event."""
    X = _make_X()
    score = EventTPR(min_offset=-2)(_events([5]), _events([4]), X)

    assert score == 1.0


@SKIP_IF_UNCHANGED
def test_event_tpr_miss():
    """An alarm outside the window does not hit the event."""
    X = _make_X()
    score = EventTPR(min_offset=-2)(_events([5]), _events([1]), X)

    assert score == 0.0


@SKIP_IF_UNCHANGED
def test_event_tpr_no_events():
    """With no true events, the score is missing."""
    X = _make_X()
    score = EventTPR(min_offset=-2)(_events([]), _events([2]), X)

    assert isinstance(score, float)
    assert np.isnan(score)


@SKIP_IF_UNCHANGED
def test_event_tpr_no_alarms():
    """With true events but no alarms, the score is 0."""
    X = _make_X()
    score = EventTPR(min_offset=-2)(_events([4, 7]), _events([]), X)

    assert score == 0.0


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize(
    "alarm, expected",
    [(7, 0.0), (8, 1.0), (11, 1.0), (12, 0.0)],
)
def test_event_tpr_window_edges(alarm, expected):
    """Alarms on both window edges hit, one step outside does not."""
    X = _make_X(n_timepoints=20)
    # event at 10, window is [8, 11]
    metric = EventTPR(min_offset=-2, max_offset=1)
    score = metric(_events([10]), _events([alarm]), X)

    assert score == expected


@SKIP_IF_UNCHANGED
def test_event_tpr_time_index():
    """Windows are in time units if the index is a time index."""
    X = _make_X(time_index=True)  # one second per step
    y_true = _events([5, 9])
    y_pred = _events([3])

    # window of the event at 5s is [3s, 5s], so only that event is hit
    score = EventTPR(min_offset=pd.Timedelta("-2s"))(y_true, y_pred, X)
    assert score == 0.5

    # a min offset of one second is too short for the alarm at 3s
    score = EventTPR(min_offset=pd.Timedelta("-1s"))(y_true, y_pred, X)
    assert score == 0.0


@SKIP_IF_UNCHANGED
def test_event_tpr_uneven_time_index():
    """Windows are measured in time, not in steps.

    Both alarms are one step before their event. The first is 10s early and
    misses the 5s window, the second is 4s early and hits it. Counting steps
    would give 1.0 instead of 0.5, so it would fail this test.
    """
    X = _make_uneven_time_X()
    y_true = _events([2, 4])  # events at 20s and 34s
    y_pred = _events([1, 3])  # alarms at 10s and 30s

    score = EventTPR(min_offset=pd.Timedelta("-5s"))(y_true, y_pred, X)

    assert score == 0.5


@SKIP_IF_UNCHANGED
def test_event_tpr_integer_index():
    """Windows use the values of an integer index, not raw positions."""
    X = pd.DataFrame({"foo": range(5)}, index=[0, 10, 20, 30, 40])
    y_true = _events([3])  # index value 30
    y_pred = _events([2])  # index value 20, one position earlier

    # the alarm is 10 index units early, so a min offset of -10 is enough
    assert EventTPR(min_offset=-10)(y_true, y_pred, X) == 1.0
    # -5 is not, even though the alarm is only one position early
    assert EventTPR(min_offset=-5)(y_true, y_pred, X) == 0.0


@SKIP_IF_UNCHANGED
def test_event_tpr_one_alarm_in_two_overlapping_windows():
    """One alarm inside two overlapping windows hits both events."""
    X = _make_X(n_timepoints=20)
    # windows are [6, 10] and [8, 12], the alarm at 9 is inside both
    y_true = _events([10, 12])
    y_pred = _events([9])

    score = EventTPR(min_offset=-4)(y_true, y_pred, X)

    assert score == 1.0


@SKIP_IF_UNCHANGED
def test_event_tpr_requires_X():
    """Without X there is no index to map positions through, so it raises."""
    metric = EventTPR(min_offset=-2)

    with pytest.raises(TypeError):
        metric(_events([5]), _events([4]))
