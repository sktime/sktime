"""Tests for the mean detection offset."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.detection._mean_detection_offset import (
    MeanDetectionOffset,
)
from sktime.tests.test_switch import run_test_for_class

SKIP_IF_UNCHANGED = pytest.mark.skipif(
    not run_test_for_class(MeanDetectionOffset),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _make_X(n_timepoints=20, time_index=False):
    """Make a series to map event positions onto."""
    if time_index:
        index = pd.date_range("2020-01-01", periods=n_timepoints, freq="s")
        return pd.DataFrame({"foo": range(n_timepoints)}, index=index)
    return pd.DataFrame({"foo": range(n_timepoints)})


def _events(ilocs):
    """Make an event table, typed like detector output, also when empty."""
    return pd.DataFrame({"ilocs": ilocs}, dtype="int64")


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_one_hit():
    """The offset is the alarm time minus the event time, so early is negative."""
    X = _make_X()
    score = MeanDetectionOffset(min_offset=-3)(_events([5]), _events([3]), X)

    assert score == -2.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_late_hit_is_positive():
    """A late hit gives a positive offset."""
    X = _make_X()
    # window is [5, 8], so the alarm 2 steps after the event hits
    score = MeanDetectionOffset(min_offset=0, max_offset=3)(
        _events([5]), _events([7]), X
    )

    assert score == 2.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_miss_is_ignored():
    """A missed event does not enter the mean."""
    X = _make_X()
    # the event at 5 is hit 2 steps early, the event at 15 is missed
    y_true = _events([5, 15])
    y_pred = _events([3])

    score = MeanDetectionOffset(min_offset=-3)(y_true, y_pred, X)

    assert score == -2.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_no_events():
    """With no true events, the score is missing."""
    X = _make_X()
    score = MeanDetectionOffset(min_offset=-3)(_events([]), _events([2]), X)

    assert isinstance(score, float)
    assert np.isnan(score)


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_no_alarms():
    """With no alarms, no event is hit, so the score is missing."""
    X = _make_X()
    score = MeanDetectionOffset(min_offset=-3)(_events([4, 7]), _events([]), X)

    assert isinstance(score, float)
    assert np.isnan(score)


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_no_hits():
    """With alarms but no hit, the score is missing."""
    X = _make_X()
    # windows are [1, 4] and [4, 7], both alarms fall outside
    score = MeanDetectionOffset(min_offset=-3)(_events([4, 7]), _events([0, 12]), X)

    assert isinstance(score, float)
    assert np.isnan(score)


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_time_index():
    """The score is a number of time units if the index is a time index."""
    X = _make_X(time_index=True)  # one second per step
    y_true = _events([5])
    y_pred = _events([2])  # 3 seconds early

    metric = MeanDetectionOffset(min_offset=pd.Timedelta("-4s"))
    assert metric(y_true, y_pred, X) == -3.0

    metric_ms = MeanDetectionOffset(min_offset=pd.Timedelta("-4s"), time_unit="ms")
    assert metric_ms(y_true, y_pred, X) == -3000.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_integer_index():
    """The score uses the values of an integer index, not raw positions."""
    X = pd.DataFrame({"foo": range(5)}, index=[0, 10, 20, 30, 40])
    y_true = _events([3])  # index value 30
    y_pred = _events([1])  # index value 10, two positions earlier

    score = MeanDetectionOffset(min_offset=-25)(y_true, y_pred, X)

    assert score == -20.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_two_hits_take_earliest():
    """With two hits for one event, the earliest alarm sets the offset."""
    X = _make_X()
    # window is [6, 10], alarms are out of order on purpose
    y_true = _events([10])
    y_pred = _events([9, 7])

    score = MeanDetectionOffset(min_offset=-4)(y_true, y_pred, X)

    assert score == -3.0


@SKIP_IF_UNCHANGED
def test_mean_detection_offset_requires_X():
    """Without X there is no index to map positions through, so it raises."""
    metric = MeanDetectionOffset(min_offset=-3)

    with pytest.raises(TypeError):
        metric(_events([5]), _events([3]))
