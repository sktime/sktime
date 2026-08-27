"""Tests for the FlexiblePanelSplitter splitter composition."""

__author__ = ["Garve"]

import numpy as np
import pandas as pd
import pytest

from sktime.split import SlidingWindowSplitter
from sktime.split.compose import FlexiblePanelSplitter
from sktime.utils._testing.series import _make_series


def _panel(lengths: dict) -> pd.DataFrame:
    """Build a panel where each instance is a suffix of a shared 0..9 timeline."""
    series = {
        name: pd.Series(np.arange(start, 10), index=pd.RangeIndex(start, 10))
        for name, start in lengths.items()
    }
    return pd.concat(series, names=["instance", "time"]).to_frame("value")


def test_flexible_panel_splitter_excludes_instances_without_full_window():
    """Folds should only contain instances that have started by the cutoff.

    This is the scenario from the feature request: an early fold, before the
    shorter instance has enough history, should contain only the longer
    instance; later folds, once the shorter instance has caught up, should
    contain both.
    """
    y = _panel({"long": 0, "short": 3})
    cv = FlexiblePanelSplitter(
        SlidingWindowSplitter(window_length=3, fh=1, step_length=3)
    )

    splits = list(cv.split_series(y))
    assert len(splits) == 3

    instances_per_fold = [set(train.index.get_level_values(0)) for train, _ in splits]
    assert instances_per_fold[0] == {"long"}
    assert instances_per_fold[1] == {"long", "short"}
    assert instances_per_fold[2] == {"long", "short"}

    # the shared fold uses each instance's own train/test window
    train1, test1 = splits[1]
    assert list(train1.loc["long"].index) == [3, 4, 5]
    assert list(test1.loc["long"].index) == [6]
    assert list(train1.loc["short"].index) == [3, 4, 5]
    assert list(test1.loc["short"].index) == [6]


def test_flexible_panel_splitter_min_length_relaxes_training_window():
    """min_length should include instances with a partial training window."""
    y = _panel({"long": 0, "medium": 2, "short": 3})
    base_cv = SlidingWindowSplitter(window_length=3, fh=1, step_length=3)

    cv_strict = FlexiblePanelSplitter(base_cv)
    first_fold_instances = set(
        next(cv_strict.split_series(y))[0].index.get_level_values(0)
    )
    assert "medium" not in first_fold_instances

    cv_relaxed = FlexiblePanelSplitter(base_cv, min_length=1)
    train0, _ = next(cv_relaxed.split_series(y))
    assert "medium" in set(train0.index.get_level_values(0))
    assert list(train0.loc["medium"].index) == [2]  # only 1 point available


def test_flexible_panel_splitter_invalid_min_length_raises():
    with pytest.raises(ValueError, match="min_length"):
        FlexiblePanelSplitter(SlidingWindowSplitter(), min_length=0)


def test_flexible_panel_splitter_min_length_above_window_warns():
    """min_length above window_length has no relaxing effect, so it should warn."""
    with pytest.warns(UserWarning, match="min_length"):
        FlexiblePanelSplitter(
            SlidingWindowSplitter(window_length=3, fh=1), min_length=5
        )


def test_flexible_panel_splitter_passthrough_for_single_series():
    """FlexiblePanelSplitter on a plain (non-panel) series should be a no-op."""
    y = _make_series(n_timepoints=20)
    base_cv = SlidingWindowSplitter(window_length=5, fh=1)
    cv = FlexiblePanelSplitter(base_cv)

    expected = list(base_cv.split(y))
    actual = list(cv.split(y))

    assert len(actual) == len(expected)
    for (train, test), (expected_train, expected_test) in zip(actual, expected):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
