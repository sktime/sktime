import numpy as np
from sklearn.metrics import auc

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils import (
    _compute_window_indices,
    _improved_cardinality_fn,
    _ts_precision_and_recall,
)

__author__ = ["Ankit-1204"]
__all__ = ["TimeSeriesAUPRC"]


def _ts_auprc(y_true, y_pred, integration="trapezoid", weighted_precision=True):
    thresholds = np.unique(y_pred)
    precision = np.empty(len(thresholds) + 1)
    recall = np.empty(len(thresholds) + 1)
    predictions = np.empty_like(y_pred, dtype=int)

    precision[-1] = 1
    recall[-1] = 0
    label_ranges = _compute_window_indices(y_true)
    for i, t in enumerate(thresholds):
        predictions = y_pred >= t
        prec, rec = _ts_precision_and_recall(
            y_true,
            predictions,
            alpha=0,
            recall_cardinality_fn=_improved_cardinality_fn,
            anomaly_ranges=label_ranges,
            weighted_precision=weighted_precision,
        )
        precision[i] = prec
        recall[i] = rec
    if integration == "riemann":
        area = -np.sum(np.diff(recall) * precision[:-1])
    else:
        area = auc(recall, precision)

    return area


class TimeSeriesAUPRC(BaseDetectionMetric):
    """Compute the area under the precision-recall curve for time series."""

    def __init__(
        self, integration="trapezoid", weighted_precision=True, with_scores=False
    ):
        self.integration = integration
        self.weighted_precision = weighted_precision
        self.with_scores = with_scores
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        self._integration = self.integration
        self._weighted_precision = self.weighted_precision
        self._with_scores = self.with_scores
        if not self._with_scores:
            n_timepoints = max(max(y_pred["ilocs"]), max(y_true["ilocs"])) + 1
            scores = np.zeros(n_timepoints)
            for pred_idx in y_pred["ilocs"]:
                distances = np.abs(pred_idx - y_true["ilocs"])
                scores[pred_idx] = 1 / (1 + np.min(distances))
            true_events = np.zeros(n_timepoints, dtype=bool)
            true_events[y_true["ilocs"]] = True
        else:
            true_events = y_true
            scores = y_pred
        return _ts_auprc(
            true_events,
            scores,
            integration=self._integration,
            weighted_precision=self._weighted_precision,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param1 = {}
        param2 = {"integration": "trapezoid", "weighted_precision": True}
        param3 = {"integration": "riemann", "weighted_precision": True}

        return [param1, param2, param3]
