import pandas as pd

from sktime.performance_metrics.detection import RandIndex


def test_randindex_identical_segments():
    # two identical segmentations -> RandIndex == 1.0
    y_true = pd.DataFrame({"ilocs": [pd.Interval(0, 5), pd.Interval(5, 10)]})
    y_pred = pd.DataFrame({"ilocs": [pd.Interval(0, 5), pd.Interval(5, 10)]})
    metric = RandIndex()
    score = metric(y_true=y_true, y_pred=y_pred, X=None)
    assert score == 1.0


def test_randindex_nontrivial_segments():
    # partial overlap -> score < 1.0 but between 0 and 1
    y_true = pd.DataFrame({"ilocs": [pd.Interval(0, 5), pd.Interval(6, 10)]})
    y_pred = pd.DataFrame({"ilocs": [pd.Interval(0, 5), pd.Interval(5, 10)]})
    metric = RandIndex()
    score = metric(y_true=y_true, y_pred=y_pred, X=None)
    assert 0.0 <= score <= 1.0
    assert score < 1.0
