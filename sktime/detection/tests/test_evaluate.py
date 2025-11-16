import pandas as pd

from sktime.detection.dummy import ZeroAnomalies
from sktime.detection.model_evaluation import evaluate
from sktime.split import SingleWindowSplitter
from sktime.performance_metrics.detection import WindowedF1Score


def test_evaluate_zero_anomalies_empty_truth():
    X = pd.Series(range(10))
    y = pd.DataFrame({"ilocs": []})

    det = ZeroAnomalies()
    cv = SingleWindowSplitter(fh=1, window_length=5)

    metric = WindowedF1Score()
    res = evaluate(detector=det, cv=cv, X=X, y=y, scoring=metric, return_data=True)

    col = f"test_{metric.name}"
    assert col in res.columns
    # With empty ground truth and empty prediction, F1 by convention == 1.0
    assert res.iloc[0][col] == 1.0


def test_evaluate_update_strategy():
    X = pd.Series(range(15))
    y = pd.DataFrame({"ilocs": []})

    det = ZeroAnomalies()
    cv = SingleWindowSplitter(fh=1, window_length=5)

    metric = WindowedF1Score()
    res = evaluate(detector=det, cv=cv, X=X, y=y, scoring=metric, return_data=False, strategy="update")

    col = f"test_{metric.name}"
    assert col in res.columns


def test_evaluate_loc_based_ilocs():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    X = pd.Series(range(10), index=idx)
    # ground truth using loc values (timestamps) rather than integer ilocs
    y = pd.DataFrame({"ilocs": [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-07")]})

    det = ZeroAnomalies()
    cv = SingleWindowSplitter(fh=1, window_length=5)

    metric = WindowedF1Score()
    res = evaluate(detector=det, cv=cv, X=X, y=y, scoring=metric, return_data=True)

    col = f"test_{metric.name}"
    assert col in res.columns
    assert isinstance(res.iloc[0][col], (int, float))
