"""Sliding window z-score and quantile based anomaly detection."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class SlidingWindowAnomalyDetector(BaseDetector):
    """Sliding window anomaly detector.

    Detects anomalies based on how far a point deviates from the
    distribution of recent values in a sliding window.

    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window.
    method : str, default="zscore"
        Method to use: "zscore" or "quantile".
    threshold : float, default=3.0
        Threshold for detection.
        For "zscore": number of standard deviations (e.g., 3.0)
        For "quantile": percentile (e.g., 95 for 95th percentile)

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.sliding_window import SlidingWindowAnomalyDetector
    >>> y = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 100, 101, 2, 3])
    >>> d = SlidingWindowAnomalyDetector(window_size=5, threshold=3.0)
    >>> y_pred = d.fit_predict(y)
    """

    _tags = {
        "authors": ["soniya-malviy"],
        "maintainers": ["soniya-malviy"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, window_size=10, method="zscore", threshold=3.0):
        self.window_size = window_size
        self.method = method
        self.threshold = threshold
        super().__init__()

        if method not in ["zscore", "quantile"]:
            raise ValueError(f"method must be 'zscore' or 'quantile', got {method}")

    def _predict(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                all_anomaly_ilocs = []
                for col in X.columns:
                    col_ilocs = self._predict_single_column(X[col].values)
                    all_anomaly_ilocs.extend(col_ilocs)
                if len(all_anomaly_ilocs) == 0:
                    return self._empty_sparse()
                return pd.DataFrame({"ilocs": sorted(set(all_anomaly_ilocs))})
            X = X.values.flatten()

        return self._predict_single_column(X)

    def _predict_single_column(self, X):
        n = len(X)
        scores = np.full(n, np.nan)
        for i in range(self.window_size, n):
            window = X[i - self.window_size : i]
            if self.method == "zscore":
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:
                    scores[i] = abs(X[i] - mean) / std
            elif self.method == "quantile":
                q = np.percentile(window, self.threshold)
                scores[i] = 1 if X[i] > q else 0
        if self.method == "zscore":
            anomaly_ilocs = np.where(scores > self.threshold)[0]
        else:
            anomaly_ilocs = np.where(scores == 1)[0]
        if len(anomaly_ilocs) == 0:
            return []
        return anomaly_ilocs.tolist()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params0 = {"window_size": 5, "threshold": 2.0}
        params1 = {"window_size": 10, "method": "quantile", "threshold": 95}
        params2 = {"window_size": 3, "threshold": 2.5}
        return [params0, params1, params2]
