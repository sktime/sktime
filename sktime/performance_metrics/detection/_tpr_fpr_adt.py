# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Event detection metrics: TPR, FPR and advance detection time.

This module implements three metrics specifically designed for evaluating
event/anomaly detectors on time series:

* ``WindowedTPR`` - True Positive Rate (Recall) with margin-based matching
* ``WindowedFPR`` - False Positive Rate with margin-based matching
* ``EarlyDetectionTime`` - Advance detection time: how early a detector fires
  relative to the true event onset

These metrics are central to the ESoC 2026 agricultural-machinery foreign-object
detection project, which requires transparent evaluation in terms of TPR, FPR and
advance detection time.

References
----------
.. [1] Talagala, P. D., Hyndman, R. J., & Smith-Miles, K. (2021).
   Anomaly detection in high-dimensional data. Journal of Computational and
   Graphical Statistics, 30(2), 360-374.
"""

__author__ = ["rupeshca007"]
__all__ = ["WindowedTPR", "WindowedFPR", "EarlyDetectionTime"]

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class WindowedTPR(BaseDetectionMetric):
    """True Positive Rate (Recall) for event detection with a tolerance window.

    Computes the fraction of true events that have at least one predicted event
    within ``margin`` time steps (iloc positions).

    A true event at iloc ``t`` is considered *detected* if any predicted event
    falls in the closed interval ``[t - margin, t + margin]``.

    Each true event can be claimed by at most one predicted event (greedy
    left-to-right matching), preventing a single prediction from satisfying
    multiple ground-truth events.

    .. math::
        \\text{TPR} = \\frac{|\\text{matched true events}|}{|\\text{true events}|}

    Parameters
    ----------
    margin : int, default=0
        Symmetric tolerance in iloc units.  A margin of 0 requires exact matches.
        A margin of ``k`` allows a predicted event to be up to ``k`` steps away
        from the true event.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._tpr_fpr_adt import WindowedTPR
    >>> y_true = pd.DataFrame({"ilocs": [10, 50, 90]})
    >>> y_pred = pd.DataFrame({"ilocs": [11, 52, 95]})
    >>> WindowedTPR(margin=5)(y_true, y_pred)
    1.0
    >>> WindowedTPR(margin=0)(y_true, y_pred)
    0.0

    See Also
    --------
    WindowedFPR : False Positive Rate counterpart.
    EarlyDetectionTime : Measures how early detections are made.
    WindowedF1Score : F1 score combining precision and recall.
    """

    _tags = {
        "authors": ["rupeshca007"],
        "maintainers": ["rupeshca007"],
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # higher TPR is better
    }

    def __init__(self, margin=0):
        self.margin = margin
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute TPR under margin-based event matching.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events with column ``"ilocs"`` (integer iloc indices).
        y_pred : pd.DataFrame
            Predicted events with column ``"ilocs"`` (integer iloc indices).
        X : pd.DataFrame, optional
            Not used; kept for API compatibility.

        Returns
        -------
        float
            TPR in ``[0, 1]``.  Returns ``1.0`` when both ``y_true`` and
            ``y_pred`` are empty (no events to miss), and ``0.0`` when
            ``y_true`` is non-empty but ``y_pred`` is empty.
        """
        gt = sorted(y_true["ilocs"].values.tolist())
        pred = sorted(y_pred["ilocs"].values.tolist())

        if len(gt) == 0 and len(pred) == 0:
            return 1.0
        if len(gt) == 0:
            return 1.0  # no ground-truth events → nothing to miss
        if len(pred) == 0:
            return 0.0  # missed everything

        margin = self.margin
        matched_gt = set()
        used_pred = set()

        for j, t in enumerate(gt):
            for i, p in enumerate(pred):
                if i in used_pred:
                    continue
                if abs(p - t) <= margin:
                    matched_gt.add(j)
                    used_pred.add(i)
                    break  # each gt event matched by at most one prediction

        return float(len(matched_gt) / len(gt))

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
        return [{}, {"margin": 3}, {"margin": 10}]


class WindowedFPR(BaseDetectionMetric):
    """False Positive Rate for event detection with a tolerance window.

    Computes the fraction of predicted events that do *not* match any true event
    within ``margin`` iloc steps, normalised by the total number of predicted
    events.

    .. math::
        \\text{FPR} = \\frac{|\\text{unmatched predictions}|}{|\\text{predictions}|}

    .. note::
        This definition treats FPR as *false discovery rate among predictions*,
        which is the most natural formulation when the number of true negatives
        (normal time steps) is very large relative to events — exactly the case
        in anomaly / rare-event detection scenarios.

    Parameters
    ----------
    margin : int, default=0
        Symmetric tolerance in iloc units.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._tpr_fpr_adt import WindowedFPR
    >>> y_true = pd.DataFrame({"ilocs": [10, 50, 90]})
    >>> y_pred = pd.DataFrame({"ilocs": [11, 30, 52]})
    >>> WindowedFPR(margin=5)(y_true, y_pred)  # 30 is a false alarm
    0.3333333333333333

    See Also
    --------
    WindowedTPR : True Positive Rate counterpart.
    EarlyDetectionTime : Measures how early detections are made.
    """

    _tags = {
        "authors": ["rupeshca007"],
        "maintainers": ["rupeshca007"],
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,  # lower FPR is better
    }

    def __init__(self, margin=0):
        self.margin = margin
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute FPR under margin-based event matching.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events with column ``"ilocs"``.
        y_pred : pd.DataFrame
            Predicted events with column ``"ilocs"``.
        X : pd.DataFrame, optional
            Not used; kept for API compatibility.

        Returns
        -------
        float
            FPR in ``[0, 1]``.  Returns ``0.0`` when ``y_pred`` is empty.
        """
        gt = sorted(y_true["ilocs"].values.tolist())
        pred = sorted(y_pred["ilocs"].values.tolist())

        if len(pred) == 0:
            return 0.0  # no predictions → no false positives
        if len(gt) == 0:
            return 1.0  # all predictions are false alarms

        margin = self.margin
        used_gt = set()
        true_positives = 0

        for p in pred:
            for j, t in enumerate(gt):
                if j in used_gt:
                    continue
                if abs(p - t) <= margin:
                    true_positives += 1
                    used_gt.add(j)
                    break

        false_positives = len(pred) - true_positives
        return float(false_positives / len(pred))

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
        return [{}, {"margin": 3}, {"margin": 10}]


class EarlyDetectionTime(BaseDetectionMetric):
    """Advance detection time: how many steps *before* the true event a detector fires.

    For each matched true event, this metric measures the signed difference:

    .. math::
        \\Delta t = t_{\\text{true}} - t_{\\text{pred}}

    * A **positive** value means the detector fired *before* the true event
      (early detection, desirable in safety-critical applications).
    * A **negative** value means the detector fired *after* the true event (late).
    * Zero means exact match.

    By default (``aggregate="mean"``), the metric returns the mean advance
    detection time over all matched events.  Unmatched true events are excluded
    from the average but counted as missed (and optionally penalised via
    ``missed_penalty``).

    Parameters
    ----------
    margin : int, default=0
        Symmetric tolerance window (in iloc units) for matching predictions to
        true events.  A prediction must fall within ``[t - margin, t + margin]``
        to be considered a match.
    aggregate : str, {"mean", "median", "min", "max"}, default="mean"
        How to aggregate advance times across matched events.
    missed_penalty : int or None, default=None
        Penalty value applied for each unmatched (missed) true event.
        If ``None``, missed events are excluded from the aggregate.
        Setting a large negative value (e.g., ``-margin``) penalises misses.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._tpr_fpr_adt import EarlyDetectionTime
    >>> y_true = pd.DataFrame({"ilocs": [20, 60, 100]})
    >>> y_pred = pd.DataFrame({"ilocs": [15, 58, 102]})  # first two early, last late
    >>> EarlyDetectionTime(margin=10)(y_true, y_pred)
    1.0

    See Also
    --------
    WindowedTPR : True Positive Rate.
    WindowedFPR : False Positive Rate.
    """

    _tags = {
        "authors": ["rupeshca007"],
        "maintainers": ["rupeshca007"],
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,  # higher advance time is better (earlier detection)
    }

    def __init__(self, margin=0, aggregate="mean", missed_penalty=None):
        self.margin = margin
        self.aggregate = aggregate
        self.missed_penalty = missed_penalty
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute mean advance detection time.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events with column ``"ilocs"``.
        y_pred : pd.DataFrame
            Predicted events with column ``"ilocs"``.
        X : pd.DataFrame, optional
            Not used; kept for API compatibility.

        Returns
        -------
        float
            Aggregated advance detection time (in iloc steps).
            Positive means early detection; negative means late.
            Returns ``np.nan`` when there are no ground-truth events.
        """
        gt = sorted(y_true["ilocs"].values.tolist())
        pred = sorted(y_pred["ilocs"].values.tolist())

        if len(gt) == 0:
            return float("nan")

        margin = self.margin
        advance_times = []
        used_pred = set()

        for t in gt:
            best_match = None
            best_dt = None
            for i, p in enumerate(pred):
                if i in used_pred:
                    continue
                if abs(p - t) <= margin:
                    dt = t - p  # positive = early detection
                    if best_match is None or abs(dt) < abs(best_dt):
                        best_match = i
                        best_dt = dt

            if best_match is not None:
                used_pred.add(best_match)
                advance_times.append(best_dt)
            elif self.missed_penalty is not None:
                advance_times.append(self.missed_penalty)

        if len(advance_times) == 0:
            return float("nan")

        arr = np.array(advance_times, dtype=float)
        agg = self.aggregate
        if agg == "mean":
            return float(np.mean(arr))
        elif agg == "median":
            return float(np.median(arr))
        elif agg == "min":
            return float(np.min(arr))
        elif agg == "max":
            return float(np.max(arr))
        else:
            raise ValueError(
                f"aggregate must be one of 'mean', 'median', 'min', 'max'; "
                f"got {agg!r}"
            )

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
        return [
            {},
            {"margin": 5, "aggregate": "median"},
            {"margin": 10, "aggregate": "mean", "missed_penalty": -10},
        ]
