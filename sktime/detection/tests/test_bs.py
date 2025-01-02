import pandas as pd
import pytest

from sktime.detection.bs import BinarySegmentation
from sktime.tests.test_switch import run_test_for_class

__author__ = ["Alex-JG3"]


@pytest.mark.skipif(
    not run_test_for_class(BinarySegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_change_points",
    [
        (pd.Series([1, 1, 1, 1, 3, 3, 3, 3]), [3]),
        (pd.Series([1, 1, 3, 3, -1, -1, 5, 5]), [1, 3, 5]),
    ],
)
def test_find_change_points(X, expected_change_points):
    """Test the binary segmentation method for finding change points."""
    model = BinarySegmentation(threshold=1)
    change_points = model._find_change_points(X, threshold=1)
    change_points.sort()
    assert change_points == expected_change_points


@pytest.mark.skipif(
    not run_test_for_class(BinarySegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_change_points",
    [
        (pd.Series([1, 1, 1, 1, 3, 3, 3, 3]), [3]),
        (pd.Series([1, 1, 3, 3, -1, -1, 5, 5]), [1, 3, 5]),
    ],
)
def test_fit_predict(X, expected_change_points):
    """Tese the fit_predict method for binary segmentation."""
    model = BinarySegmentation(threshold=1, min_cp_distance=1)
    change_points = model.fit_predict(X)
    assert change_points.values.flatten().tolist() == expected_change_points


@pytest.mark.skipif(
    not run_test_for_class(BinarySegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_change_points,min_cp_distance",
    [
        (pd.Series([-5, -5, -5, -5, 5, 5, 3, 3]), [3, 5], 0),
        (pd.Series([-5, -5, -5, -5, 5, 5, 3, 3]), [3, 5], 1),
        (pd.Series([-5, -5, -5, -5, 5, 5, 3, 3]), [3], 2),
        (pd.Series([-5, -5, -5, -5, 5, 5, 3, 3]), [3], 3),
        (pd.Series([-5, -5, -5, -5, 5, 5, 3, 3]), [], 4),
    ],
)
def test_min_seg_length(X, expected_change_points, min_cp_distance):
    """Test the affect of change the minimum distance between change points."""
    model = BinarySegmentation(X)
    change_points = model._find_change_points(X, 1, min_cp_distance=min_cp_distance)
    change_points.sort()
    assert change_points == expected_change_points
