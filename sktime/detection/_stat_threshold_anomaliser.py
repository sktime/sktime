# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Anomaly detection by thresholding segment statistics."""

__author__ = ["Tveten"]
__all__ = ["StatThresholdAnomaliser"]

import numpy as np
import pandas as pd

from sktime.detection._formatters import format_anomaly_points
from sktime.detection.base import BaseDetector


class StatThresholdAnomaliser(BaseDetector):
    """Anomaly detection by thresholding segment statistics.

    Segments data using a change-point detector, computes a statistic per
    segment, and flags segments whose statistic falls outside a given range
    as anomalous.

    Parameters
    ----------
    change_detector : BaseDetector
        Change-point detector used to segment the data.
    stat : callable, default=np.mean
        Statistic applied per segment.  Must accept a 1-D array and return
        a scalar.
    stat_lower : float, default=-1.0
        Lower threshold — segments with ``stat < stat_lower`` are anomalous.
    stat_upper : float, default=1.0
        Upper threshold — segments with ``stat > stat_upper`` are anomalous.

    Examples
    --------
    >>> from sktime.detection._stat_threshold_anomaliser import (
    ...     StatThresholdAnomaliser,
    ... )
    >>> from sktime.detection._moving_window import MovingWindow
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> X = pd.DataFrame(rng.standard_normal((100, 1)))
    >>> X.iloc[40:60] += 10.0
    >>> detector = StatThresholdAnomaliser(
    ...     change_detector=MovingWindow(bandwidth=5),
    ...     stat_lower=-2.0,
    ...     stat_upper=2.0,
    ... )
    >>> detector.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": False,
        "fit_is_empty": True,
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
    }

    def __init__(
        self,
        change_detector,
        stat=np.mean,
        stat_lower=-1.0,
        stat_upper=1.0,
    ):
        self.change_detector = change_detector
        self.stat = stat
        self.stat_lower = stat_lower
        self.stat_upper = stat_upper
        super().__init__()

        if self.stat_lower > self.stat_upper:
            raise ValueError(
                f"stat_lower ({self.stat_lower}) must be less than or equal"
                f" to stat_upper ({self.stat_upper})."
            )

    def _fit(self, X, y=None):
        """No-op (stateless detector)."""
        return self

    def _predict(self, X):
        """Detect anomalous segments in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data (univariate).

        Returns
        -------
        pd.DataFrame
            ``"ilocs"`` column with integer point indices of anomalous
            time points.
        """
        detector = self.change_detector.clone()
        detector.fit(X)
        segments = detector.transform(X)["labels"]
        df = pd.concat([X, segments], axis=1)

        anomalies = []
        for _, segment in df.reset_index(drop=True).groupby("labels"):
            segment_stat = self.stat(segment.iloc[:, 0].values)
            if segment_stat < self.stat_lower or segment_stat > self.stat_upper:
                anomalies.append((int(segment.index[0]), int(segment.index[-1] + 1)))

        return format_anomaly_points(anomalies)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._moving_window import MovingWindow

        params = [
            {
                "change_detector": MovingWindow(bandwidth=3),
                "stat": np.mean,
                "stat_lower": -1.0,
                "stat_upper": 1.0,
            },
            {
                "change_detector": MovingWindow(bandwidth=5),
                "stat": np.median,
                "stat_lower": -2.0,
                "stat_upper": 2.0,
            },
        ]
        return params
