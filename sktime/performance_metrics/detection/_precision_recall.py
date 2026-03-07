"""Windowed Precision and Recall metrics for detection tasks."""

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils._matching import (
    _count_windowed_matches,
)


class WindowedPrecision(BaseDetectionMetric):
    """Precision for event detection, using a margin-based match criterion.

    Computes iloc-based Precision with a margin of error.
    A predicted event is considered a true positive if there is a ground truth
    event within the specified ``margin`` (in iloc units).

    Precision = TP / (TP + FP), i.e., of all predicted events,
    what fraction are close to a true event.

    Parameters
    ----------
    margin : int, optional (default=0)
        Margin of error in iloc units. A predicted event matches a ground truth
        event if their absolute iloc difference is <= margin.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import WindowedPrecision
    >>> y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
    >>> y_pred = pd.DataFrame({"ilocs": [2, 4, 9]})
    >>> metric = WindowedPrecision(margin=1)
    >>> metric(y_true, y_pred)
    1.0
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,
    }

    def __init__(self, margin=0):
        self.margin = margin
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute Precision under margin-based breakpoint matching logic.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth breakpoints in "points" format. Must have column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted breakpoints in "points" format. Must have column 'ilocs'.
        X : pd.DataFrame, optional (default=None)
            Unused, but part of the signature.

        Returns
        -------
        float
            Precision score = TP / (TP + FP).
        """
        gt = sorted(y_true["ilocs"].values)
        pred = sorted(y_pred["ilocs"].values)
        margin = self.margin

        # Edge cases
        if len(pred) == 0 and len(gt) == 0:
            return 1.0
        if len(pred) == 0:
            return 0.0
        if len(gt) == 0:
            return 0.0

        matched_count = _count_windowed_matches(gt, pred, margin)
        return matched_count / len(pred)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict
        """
        param0 = {}
        param1 = {"margin": 1}
        param2 = {"margin": 5}
        return [param0, param1, param2]


class WindowedRecall(BaseDetectionMetric):
    """Recall for event detection, using a margin-based match criterion.

    Computes iloc-based Recall with a margin of error.
    A ground truth event is considered detected if there is a predicted event
    within the specified ``margin`` (in iloc units).

    Recall = TP / (TP + FN), i.e., of all true events,
    what fraction were detected.

    Parameters
    ----------
    margin : int, optional (default=0)
        Margin of error in iloc units. A predicted event matches a ground truth
        event if their absolute iloc difference is <= margin.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import WindowedRecall
    >>> y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
    >>> y_pred = pd.DataFrame({"ilocs": [2, 4, 9]})
    >>> metric = WindowedRecall(margin=1)
    >>> metric(y_true, y_pred)
    1.0
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,
    }

    def __init__(self, margin=0):
        self.margin = margin
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute Recall under margin-based breakpoint matching logic.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth breakpoints in "points" format. Must have column 'ilocs'.
        y_pred : pd.DataFrame
            Predicted breakpoints in "points" format. Must have column 'ilocs'.
        X : pd.DataFrame, optional (default=None)
            Unused, but part of the signature.

        Returns
        -------
        float
            Recall score = TP / (TP + FN).
        """
        gt = sorted(y_true["ilocs"].values)
        pred = sorted(y_pred["ilocs"].values)
        margin = self.margin

        # Edge cases
        if len(gt) == 0 and len(pred) == 0:
            return 1.0
        if len(gt) == 0:
            return 0.0
        if len(pred) == 0:
            return 0.0

        matched_count = _count_windowed_matches(gt, pred, margin)
        return matched_count / len(gt)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict
        """
        param0 = {}
        param1 = {"margin": 1}
        param2 = {"margin": 5}
        return [param0, param1, param2]
