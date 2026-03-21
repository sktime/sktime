"""Anomaly detectors composed of change detectors and some conversion logic."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..change_detectors.base import BaseChangeDetector
from .base import BaseSegmentAnomalyDetector


class StatThresholdAnomaliser(BaseSegmentAnomalyDetector):
    """Anomaly detection based on thresholding the values of segment statistics.

    Parameters
    ----------
    change_detector : BaseChangeDetector
        Change detector to use for detecting segments.
    stat : callable, optional (default=np.mean)
        Statistic to calculate per segment. A function that takes in a 1D array and
        returns a float.
    stat_lower : float, optional (default=-1.0)
        Segments with a statistic lower than this value are considered anomalous.
    stat_upper : float, optional (default=1.0)
        Segments with a statistic higher than this value are considered anomalous.
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        change_detector: BaseChangeDetector,
        stat: Callable[[np.ndarray], float] = np.mean,
        stat_lower: float = -1.0,
        stat_upper: float = 1.0,
    ):
        self.change_detector = change_detector
        self.stat = stat
        self.stat_lower = stat_lower
        self.stat_upper = stat_upper
        super().__init__()

        if self.stat_lower > self.stat_upper:
            message = (
                f"stat_lower ({self.stat_lower}) must be less"
                + f" than or equal to stat_upper ({self.stat_upper})."
            )
            raise ValueError(message)

        self.clone_tags(change_detector, ["distribution_type"])

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit the detector to.
        y : pd.Series, optional
            Does nothing. Only here to make the fit method compatible with `sktime`
            and `scikit-learn`.

        Returns
        -------
        self :
            Reference to self.

        State change
        ------------
        Creates fitted model that updates attributes ending in "_".
        """
        self.change_detector_: BaseChangeDetector = self.change_detector.clone()
        self.change_detector_.fit(X, y)
        return self

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect anomalies in.

        Returns
        -------
        y_sparse: pd.DataFrame
            A `pd.DataFrame` with a range index and two columns:
            * ``"ilocs"`` - left-closed ``pd.Interval``s of iloc based segments.
            * ``"labels"`` - integer labels ``1, ..., K`` for each segment anomaly.
        """
        # This is the required output format for the rest of the code to work.
        segments = self.change_detector_.transform(X)["labels"]
        df = pd.concat([X, segments], axis=1)
        anomalies = []
        for _, segment in df.reset_index(drop=True).groupby("labels"):
            segment_stat = self.stat(segment.iloc[:, 0].values)
            if (segment_stat < self.stat_lower) | (segment_stat > self.stat_upper):
                anomalies.append((int(segment.index[0]), int(segment.index[-1] + 1)))

        return self._format_sparse_output(anomalies)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            MyClass(**params) or MyClass(**params[i]) creates a valid test instance.
            create_test_instance uses the first (or only) dictionary in params
        """
        from sktime.detection._skchange.change_detectors import MovingWindow

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
