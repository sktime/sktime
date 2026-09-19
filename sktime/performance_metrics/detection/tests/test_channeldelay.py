"""Tests for MultiChannelDetectionDelay."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection import MultiChannelDetectionDelay


def _frames(n_channels, true_locs, pred_locs):
    """Build y_true / y_pred from plain lists — channels named ch_0, ch_1, ..."""
    cols = [f"ch_{i}" for i in range(n_channels)]
    y_true = pd.DataFrame(dict(zip(cols, true_locs)))
    y_pred = pd.DataFrame(dict(zip(cols, pred_locs)))
    return cols, y_true, y_pred


def test_all_channels_early_mean():
    # every channel fires before the true event → mean delay is 0
    cols, y_true, y_pred = _frames(3, [[500], [500], [500]], [[480], [490], [495]])
    metric = MultiChannelDetectionDelay(channel_cols=cols, aggfunc="mean")
    assert metric(y_true, y_pred) == 0.0


def test_min_one_channel_enough():
    # ch_0 is early, ch_2 is late — min should still return 0
    cols, y_true, y_pred = _frames(3, [[500], [500], [500]], [[480], [500], [510]])
    metric = MultiChannelDetectionDelay(
        channel_cols=cols, aggfunc="min", early_tolerance=20, max_delay=50
    )
    assert metric(y_true, y_pred) == 0.0


def test_max_surfaces_slowest_channel():
    cols, y_true, y_pred = _frames(3, [[500], [500], [500]], [[490], [495], [520]])
    metric = MultiChannelDetectionDelay(channel_cols=cols, aggfunc="max")
    assert metric(y_true, y_pred) == 20.0


def test_weighted_aggfunc():
    # delays: ch_0=0, ch_1=10, ch_2=20
    # weights: 0.5, 0.3, 0.2  →  0*0.5 + 10*0.3 + 20*0.2 = 7.0
    cols, y_true, y_pred = _frames(3, [[500], [500], [500]], [[500], [510], [520]])
    metric = MultiChannelDetectionDelay(
        channel_cols=cols,
        aggfunc="weighted",
        channel_weights=[0.5, 0.3, 0.2],
    )
    assert metric(y_true, y_pred) == pytest.approx(7.0)


def test_arbitrary_channel_count():
    # works for any N — here we try 6 to show it's not hardcoded to 3
    cols, y_true, y_pred = _frames(
        6,
        [[500]] * 6,
        [[500 + i * 5] for i in range(6)],  # delays: 0, 5, 10, 15, 20, 25
    )
    metric = MultiChannelDetectionDelay(channel_cols=cols, aggfunc="mean")
    assert metric(y_true, y_pred) == pytest.approx(12.5)


def test_multiple_events_per_channel():
    # two events per channel — checks the inner loop doesn't break
    cols, y_true, y_pred = _frames(
        2,
        [[200, 500], [200, 500]],
        [[190, 490], [205, 510]],
    )
    metric = MultiChannelDetectionDelay(
        channel_cols=cols, aggfunc="mean", early_tolerance=15, max_delay=50
    )
    score = metric(y_true, y_pred)
    assert score >= 0.0


def test_no_channel_cols_raises():
    _, y_true, y_pred = _frames(2, [[500], [500]], [[490], [490]])
    metric = MultiChannelDetectionDelay()
    with pytest.raises(ValueError, match="channel_cols cannot be None"):
        metric(y_true, y_pred)


def test_weighted_without_weights_raises():
    cols, y_true, y_pred = _frames(2, [[500], [500]], [[490], [490]])
    metric = MultiChannelDetectionDelay(channel_cols=cols, aggfunc="weighted")
    with pytest.raises(ValueError, match="channel_weights must be set"):
        metric(y_true, y_pred)
