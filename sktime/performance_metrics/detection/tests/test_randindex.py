"""Tests for the segmentation Rand index."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import rand_score

from sktime.performance_metrics.detection import RandIndex
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(RandIndex),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _segments(breaks, labels=None):
    """Build a sparse segmentation with optional cluster labels."""
    result = pd.DataFrame(
        {"ilocs": pd.IntervalIndex.from_breaks(breaks, closed="left")}
    )
    if labels is not None:
        result["label"] = labels
    return result


@pytest.mark.parametrize(
    "true_breaks, pred_breaks, true_labels, pred_labels",
    [
        ([0, 3, 6], [0, 3, 6], [0, 1], [10, 20]),
        ([0, 3, 6], [0, 3, 6], [0, 1], [1, 0]),
        ([0, 2, 4, 6], [0, 3, 6], None, None),
        ([0, 6], [0, 2, 4, 6], None, None),
        ([0, 2, 4, 6], [0, 3, 6], [0, 1, 0], [10, 20]),
        ([0, 2, 4, 6], [0, 2, 4, 6], [0, 1, 0], [10, 20, 10]),
        ([0, 6], [0, 2, 4, 6], [0], [10, 10, 10]),
        ([0, 3, 6], [0, 2, 6], ["a", "b"], ["x", "y"]),
    ],
    ids=[
        "renamed-labels",
        "permuted-labels",
        "different-segment-counts",
        "single-vs-multiple-segments",
        "repeated-true-label",
        "repeated-label-pair",
        "subdivided-cluster",
        "string-labels",
    ],
)
def test_randindex_matches_clustering(
    true_breaks, pred_breaks, true_labels, pred_labels
):
    """Sparse segments must give the same Rand score as dense cluster labels."""
    y_true = _segments(true_breaks, true_labels)
    y_pred = _segments(pred_breaks, pred_labels)
    if true_labels is None:
        true_labels = range(len(true_breaks) - 1)
    if pred_labels is None:
        pred_labels = range(len(pred_breaks) - 1)
    dense_true = np.repeat(true_labels, np.diff(true_breaks))
    dense_pred = np.repeat(pred_labels, np.diff(pred_breaks))
    expected = rand_score(dense_true, dense_pred)

    metric = RandIndex(use_loc=False)
    assert metric(y_true, y_pred) == pytest.approx(expected)
    assert metric(y_pred, y_true) == pytest.approx(expected)


def test_randindex_loc_label_invariance():
    """Renaming labels must also preserve scores with irregular loc distances."""
    X = pd.DataFrame({"value": range(5)}, index=[0, 2, 5, 9, 12])
    y_true = _segments([0, 2, 5], [0, 1])
    y_pred = _segments([0, 2, 5], [10, 20])

    metric = RandIndex(use_loc=True)
    assert metric(y_true, y_true, X) == 1.0
    assert metric(y_true, y_pred, X) == 1.0
