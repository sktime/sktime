import pandas as pd
import pytest

from sktime.annotation.bs import BinarySegmentation


@pytest.mark.parametrize(
    "X,expected_change_points",
    [
        (pd.Series([1, 1, 1, 1, 3, 3, 3, 3]), [3]),
        (pd.Series([1, 1, 3, 3, -1, -1, 5, 5]), [1, 3, 5]),
    ],
)
def test_find_change_points(X, expected_change_points):
    change_points = []
    model = BinarySegmentation(X)
    model._find_change_points(X, 0, len(X) - 1, 1, change_points)
    change_points.sort()
    assert tuple(change_points) == tuple(expected_change_points)
