import pandas as pd

from sktime.performance_metrics.detection import DetectionDelayMean


def test_mean_delay():
    metric = DetectionDelayMean()

    y_true = pd.DataFrame({"ilocs": [100, 200]})

    # perfect match
    y_pred = pd.DataFrame({"ilocs": [100, 200]})
    assert metric(y_true, y_pred) == 0.0

    # delayed
    y_pred = pd.DataFrame({"ilocs": [110, 210]})
    assert metric(y_true, y_pred) == 10.0

    # early detection (should not penalize)
    y_pred = pd.DataFrame({"ilocs": [90, 190]})
    assert metric(y_true, y_pred) == 0.0
