"""Mean shift change point detector based on Moving Averages."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector

__author__ = ["Waqibsk"]


class MeanShift(BaseDetector):
    """Mean Shift (Moving Average) change point detector.

    This detector identifies change points by comparing the moving average of a
    trailing window (left) against a leading window (right). If the absolute
    difference between these means exceeds a specified threshold, a change point
    is flagged.

    Parameters
    ----------
    window_size : int
        The size of the sliding window used to calculate the means on either side
        of a potential change point.
    threshold : float
        The threshold for the difference between the left and right window means.
        If the difference is greater than this value, it is considered a candidate
        change point.
    min_cp_distance : int, default=0
        Minimum distance between detected change points. If multiple points exceed
        the threshold in close proximity, only the one with the highest divergence
        is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.ma import MeanShift
    >>> X = pd.Series([1, 1, 1, 1, 5, 5, 5, 5])
    >>> model = MeanShift(window_size=2, threshold=2)
    >>> model.fit_predict(X)
       ilocs
    0    3
    dtype: int64
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Waqibsk"],
        "maintainers": ["Waqibsk"],
        # estimator type
        # --------------
        "fit_is_empty": True,
        "capability:multivariate": False,
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.Series",
    }

    def __init__(self, window_size, threshold, min_cp_distance=0):
        self.window_size = window_size
        self.threshold = threshold
        self.min_cp_distance = min_cp_distance
        super().__init__()

    def _compute_scores(self, X):
        """Compute the divergence scores between left and right windows.

        Parameters
        ----------
        X : pd.Series
            The input time series.

        Returns
        -------
        pd.Series
            A series of absolute differences between left and right rolling means.
        """
        # compute left rolling mean 
        left_mean = X.rolling(window=self.window_size).mean()

        # compute right rolling mean (leading) using a reverse roll
        # we shift X so that the index aligns with the "split" point
        right_mean = (
            X.iloc[::-1]
            .rolling(window=self.window_size)
            .mean()
            .iloc[::-1]
            .shift(-1) 
        )

        # calculate absolute difference
        scores = (left_mean - right_mean).abs()
        return scores.fillna(0)

    def _find_change_points(self, scores):
        """Find peaks in the score series that exceed the threshold.

        Parameters
        ----------
        scores : pd.Series
            The divergence scores calculated by _compute_scores.

        Returns
        -------
        list[int]
            A sorted list of change point indices.
        """
        candidates = scores[scores > self.threshold]
        if candidates.empty:
            return []

        # sort candidates by score descending to prioritize strongest signals
        candidate_indices = candidates.sort_values(ascending=False).index.tolist()
        
        final_change_points = []
        
        # simple greedy suppression based on min_cp_distance
        for cp in candidate_indices:
            is_too_close = False
            for selected in final_change_points:
                if abs(cp - selected) < self.min_cp_distance:
                    is_too_close = True
                    break
            
            if not is_too_close:
                final_change_points.append(cp)

        return sorted(final_change_points)

    def _predict(self, X, Y=None):
        """Detect change points in X.

        Parameters
        ----------
        X : pd.Series
            Timeseries on which the change points will be detected.
        Y : any
            Unused argument. Included for compatibility with sklearn.

        Returns
        -------
        pd.Series
            Series whose values are the indexes of the change points.
        """
        scores = self._compute_scores(X)
        change_points = self._find_change_points(scores)
        return pd.Series(change_points, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params0 = {"window_size": 2, "threshold": 1.0}
        params1 = {"window_size": 3, "threshold": 0.5, "min_cp_distance": 2}
        return [params0, params1]