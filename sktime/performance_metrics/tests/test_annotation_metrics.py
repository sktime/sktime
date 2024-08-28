"""Tests for classes in _classes module."""

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from sktime.performance_metrics.annotation import metrics


def test_padded_f1_with_sklearn():
    """Compare padded f1 with sklearn's f1 score.

    f1 score and padded f1 score should be the same when there is one change point per
    interval.
    """
    true_change_points = pd.Series([10, 20])
    predicted_change_points = pd.Series([10])

    padded_f1 = metrics.padded_f1(true_change_points, predicted_change_points, pad=1)
    sklearn_f1 = f1_score([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    assert padded_f1 == sklearn_f1


@pytest.mark.parametrize(
    "true_cps,pred_cps,pad",
    [
        ([5, 15], [5, 14], 2),
        (
            pd.Series(["2024-01-01", "2024-06-01"], dtype="datetime64[ns]"),
            pd.Series(["2023-12-01", "2024-06-15"], dtype="datetime64[ns]"),
            pd.tseries.offsets.DateOffset(months=2),
        ),
        ([-0.5, 1.6, -5], [-0.3, 1.4, -5], 0.3),
        pytest.param([5, 15], [5, 18], 2, marks=pytest.mark.xfail),
    ],
)
def test_padded_f1_for_perfect_score(true_cps, pred_cps, pad):
    """Ensure padded F1 score returns 1.0 when all change points are detected."""
    padded_f1 = metrics.padded_f1(pd.Series(true_cps), pd.Series(pred_cps), pad)
    assert padded_f1 == 1.0


@pytest.mark.parametrize(
    "true_cps,pred_cps,threshold",
    [
        ([5, 15], [1, 10, 20], 2),
        (
            pd.Series(["2024-01-01", "2024-06-01"], dtype="datetime64[ns]"),
            pd.Series(["2023-06-01", "2024-04-01"]),
            pd.tseries.offsets.DateOffset(months=1),
        ),
        ([-0.5, 1.6], [0.0, 1.0], 0.3),
    ],
)
def test_padded_f1_for_worst_score(true_cps, pred_cps, threshold):
    """Ensure padded F1 score returns 0.0 when no of the change points are detected."""
    padded_f1 = metrics.padded_f1(pd.Series(true_cps), pd.Series(pred_cps), threshold)
    assert padded_f1 == 0.0


@pytest.mark.parametrize(
    "true_cps,pred_cps,thresh,expected_value",
    [
        ([5, 20], [5, 10], 2, 0.5),
        ([5, 12, 13], [5, 10], 1, 0.4),
        ([4, 10], [5, 12], 1, 0.5),
        ([4, 10], [5, 12], 2, 1.0),
    ],
)
def test_padded_f1_against_known_score(true_cps, pred_cps, thresh, expected_value):
    """Test padded f1 against pre-calculated tests where we know the expected value"""
    padded_f1 = metrics.padded_f1(pd.Series(true_cps), pd.Series(pred_cps), thresh)
    assert padded_f1 == expected_value
