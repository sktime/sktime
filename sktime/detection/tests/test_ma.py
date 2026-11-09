import pandas as pd
import pytest

from sktime.detection.ma import MeanShift
from sktime.tests.test_switch import run_test_for_class

__author__ = ["Waqibsk"]


@pytest.mark.skipif(
    not run_test_for_class(MeanShift),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_change_points",
    [
        (pd.Series([1, 1, 1, 1, 3, 3, 3, 3]), [3]),
    ],
)
def test_fit_predict(X, expected_change_points):
    """Test the fit_predict method for mean shift."""
    model = MeanShift(window_size=2, threshold=1)
    change_points = model.fit_predict(X)
    assert change_points.values.flatten().tolist() == expected_change_points


@pytest.mark.skipif(
    not run_test_for_class(MeanShift),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_change_points,min_cp_distance",
    [
        (pd.Series([0, 0, 0, 10, 10, 10]), [1, 2, 3], 0),
        (pd.Series([0, 0, 0, 10, 10, 10]), [2], 2),
    ],
)
def test_min_cp_distance(X, expected_change_points, min_cp_distance):
    """Test the effect of changing the minimum distance between change points."""
    model = MeanShift(window_size=2, threshold=4, min_cp_distance=min_cp_distance)
    change_points = model.fit_predict(X)
    assert sorted(change_points.values.flatten().tolist()) == expected_change_points