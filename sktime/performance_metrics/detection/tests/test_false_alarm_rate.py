"""Tests for the false alarm rate."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.detection._false_alarm_rate import FalseAlarmRate
from sktime.tests.test_switch import run_test_for_class

SKIP_IF_UNCHANGED = pytest.mark.skipif(
    not run_test_for_class(FalseAlarmRate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _make_X(n_timepoints=9, freq=None):
    """Make a series, with a time index of step ``freq`` if given."""
    if freq is not None:
        index = pd.date_range("2020-01-01", periods=n_timepoints, freq=freq)
        return pd.DataFrame({"foo": range(n_timepoints)}, index=index)
    return pd.DataFrame({"foo": range(n_timepoints)})


def _events(ilocs):
    """Make an event table, typed like detector output, also when empty."""
    return pd.DataFrame({"ilocs": ilocs}, dtype="int64")


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_unmatched_alarm():
    """An alarm that hits no event counts, over the span of X."""
    X = _make_X()  # index 0 to 8, span 8
    # window is [3, 5], so the alarm at 4 hits and the alarm at 1 does not
    score = FalseAlarmRate(min_offset=-2)(_events([5]), _events([1, 4]), X)

    assert score == 1 / 8


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_alarm_in_hit_window_is_not_false():
    """Extra alarms inside a hit window are not false alarms."""
    X = _make_X()
    # window is [3, 5], all three alarms are inside it
    score = FalseAlarmRate(min_offset=-2)(_events([5]), _events([3, 4, 5]), X)

    assert score == 0.0


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_no_events():
    """With no true events the rate is defined, every alarm is false."""
    X = _make_X()
    score = FalseAlarmRate(min_offset=-2)(_events([]), _events([2, 6]), X)

    assert score == 2 / 8


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_no_alarms():
    """With no alarms, the rate is 0."""
    X = _make_X()
    score = FalseAlarmRate(min_offset=-2)(_events([4, 7]), _events([]), X)

    assert score == 0.0


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_time_index():
    """With a time index, the rate is per time_unit, by default per hour."""
    X = _make_X(n_timepoints=7, freq="20min")  # span 120 minutes
    y_true = _events([5])  # event at 100 minutes
    y_pred = _events([1, 4])  # alarms at 20 and 80 minutes

    # window is [80min, 100min], so one false alarm in two hours
    metric = FalseAlarmRate(min_offset=pd.Timedelta("-20min"))
    assert metric(y_true, y_pred, X) == 0.5

    metric_min = FalseAlarmRate(min_offset=pd.Timedelta("-20min"), time_unit="min")
    assert metric_min(y_true, y_pred, X) == pytest.approx(1 / 120)


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_integer_index():
    """The span uses the values of an integer index, time_unit is ignored."""
    X = pd.DataFrame({"foo": range(5)}, index=[0, 10, 20, 30, 40])  # span 40
    y_true = _events([3])  # index value 30
    y_pred = _events([0, 2])  # index values 0 and 20

    # window is [20, 30], so the alarm at 0 is the one false alarm
    assert FalseAlarmRate(min_offset=-10)(y_true, y_pred, X) == 1 / 40
    # time_unit has no effect without a time index
    metric = FalseAlarmRate(min_offset=-10, time_unit="min")
    assert metric(y_true, y_pred, X) == 1 / 40


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_uneven_time_spacing():
    """Matching and span are both measured in time, not in steps.

    Both alarms are one step before their event. The first is 10s early and
    misses the 5s window, the second is 4s early and hits it. The span is
    34s, not 4 steps. Counting steps would give 0 false alarms, or a rate
    per 4 steps, so it would fail this test.
    """
    seconds = [0, 10, 20, 30, 34]
    index = pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s")
    X = pd.DataFrame({"foo": range(len(seconds))}, index=index)
    y_true = _events([2, 4])  # events at 20s and 34s
    y_pred = _events([1, 3])  # alarms at 10s and 30s

    metric = FalseAlarmRate(min_offset=pd.Timedelta("-5s"), time_unit="s")
    score = metric(y_true, y_pred, X)

    assert score == pytest.approx(1 / 34)


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_zero_span():
    """With a single time point there is no span, so the rate is missing."""
    X = _make_X(n_timepoints=1)
    score = FalseAlarmRate()(_events([]), _events([0]), X)

    assert isinstance(score, float)
    assert np.isnan(score)


@SKIP_IF_UNCHANGED
def test_false_alarm_rate_requires_X():
    """Without X there is no index to map positions through, so it raises."""
    metric = FalseAlarmRate(min_offset=-2)

    with pytest.raises(TypeError):
        metric(_events([5]), _events([1]))
