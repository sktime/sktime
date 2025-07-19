"""Tests for SubLOF"""

__author__ = ["Alex-JG3"]

import datetime

import pandas as pd
import pytest

from sktime.detection.lof import SubLOF
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(SubLOF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "x,interval_size,expected_intervals",
    [
        (
            pd.Index([1, 2, 3, 4, 5]),
            1,
            pd.IntervalIndex.from_breaks([1, 2, 3, 4, 5, 6], closed="left"),
        ),
        (
            pd.Index([0.0, 1.3, 1.5, 2.0, 3.5]),
            1.5,
            pd.IntervalIndex.from_breaks([0.0, 1.5, 3.0, 4.5], closed="left"),
        ),
        (
            pd.DatetimeIndex(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                    "2024-01-05 00:00:00",
                ]
            ),
            datetime.timedelta(days=1),
            pd.IntervalIndex.from_breaks(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                    "2024-01-05 00:00:00",
                    "2024-01-06 00:00:00",
                ],
                closed="left",
                dtype="interval[datetime64[ns], left]",
            ),
        ),
    ],
)
def test_cut_into_intervals(x, interval_size, expected_intervals):
    """Check if the predicted change points match."""
    actual_intervals = SubLOF._split_into_intervals(x, interval_size)
    assert (actual_intervals == expected_intervals).all()


@pytest.mark.skipif(
    not run_test_for_class(SubLOF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,y_expected",
    [
        (
            pd.DataFrame([0, 0, 100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 0]),
            pd.DataFrame({"ilocs": [2, 7, 13]}),
        ),
    ],
)
def test_predict(X, y_expected):
    model = SubLOF(3, window_size=5, novelty=True)
    model.fit(X)
    y_actual = model.predict(X)
    pd.testing.assert_frame_equal(y_actual, y_expected)
