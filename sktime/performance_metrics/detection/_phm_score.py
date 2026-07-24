"""PHM 2008 asymmetric scoring function for advance event detection."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class PHMScore(BaseDetectionMetric):
    r"""PHM 2008 asymmetric scoring function for advance event detection.

    The PHM Scoring Function penalizes *late* detections exponentially harder
    than *early* detections, which makes it suitable for advance / predictive
    event detection scenarios where early warning is valued and late
    detection is costly.

    For each matched pair ``(t_true, t_pred)``, the signed delay is
    ``d = t_pred - t_true`` (negative values = early, positive = late), and
    the per-pair score is

    .. math::

        s(d) = \begin{cases}
            \exp(-d / \tau_{\text{early}}) - 1 & \text{if } d \le 0 \\
            \exp( d / \tau_{\text{late}})  - 1 & \text{if } d >   0
        \end{cases}

    The default constants ``tau_early = 13`` and ``tau_late = 10`` are the
    canonical values from the PHM 2008 Prognostics Data Challenge [1]_.
    Because ``tau_late < tau_early``, the late branch grows exponentially
    faster than the early branch for the same magnitude of error, so a
    prediction that is late by k time units is penalized strictly more than
    a prediction that is early by the same k time units.

    Matching between predictions and true events is performed greedily:
    predictions are processed in time order and each prediction is matched
    to its nearest yet-unmatched true event. Unmatched true events (missed
    detections) and unmatched predictions (false alarms) are each penalized
    with ``unmatched_penalty``.

    Lower is better. A score of ``0`` means perfect detection (every
    prediction exactly at its corresponding true event, no misses, no
    false alarms).

    Parameters
    ----------
    tau_early : float, default=13.0
        Time constant for early predictions (``d <= 0``). Larger values =
        gentler penalty for being early. Default is 13, the PHM 2008
        canonical value.
    tau_late : float, default=10.0
        Time constant for late predictions (``d > 0``). Smaller values =
        harsher penalty for being late. Default is 10, the PHM 2008
        canonical value.
    unmatched_penalty : float, default=np.inf
        Per-event penalty applied to each unmatched true event (missed
        detection) and each unmatched prediction (false alarm). The
        default ``np.inf`` makes any miss or false alarm produce an
        infinite score, which matches the strict semantic of the issue
        ("no score for a detection after the event", generalized to
        "no credit for any unmatched event"). Set to a finite value
        (e.g. a large constant) if the metric is to be used for ranking
        imperfect detectors or as a training objective.
    aggregation : {"sum", "mean"}, default="sum"
        How per-event scores are combined. ``"sum"`` matches the original
        PHM 2008 scoring convention. ``"mean"`` normalizes by the total
        number of events considered (matches plus unmatched), which makes
        scores comparable across series with different numbers of events.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import PHMScore
    >>>
    >>> y_true = pd.DataFrame({"ilocs": [100, 200, 300]})
    >>> y_pred = pd.DataFrame({"ilocs": [100, 200, 300]})
    >>> PHMScore()(y_true, y_pred)
    0.0
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(
        self,
        tau_early=13.0,
        tau_late=10.0,
        unmatched_penalty=np.inf,
        aggregation="sum",
    ):
        self.tau_early = tau_early
        self.tau_late = tau_late
        self.unmatched_penalty = unmatched_penalty
        self.aggregation = aggregation
        super().__init__()

    def _asymmetric_cost(self, d):
        """Return the PHM asymmetric exponential cost for one signed delay.

        Parameters
        ----------
        d : float
            Signed delay, ``t_pred - t_true``. Negative = early, positive = late.

        Returns
        -------
        float
            Per-event cost. ``0.0`` exactly when ``d == 0``, strictly positive
            otherwise.
        """
        if d <= 0:
            return float(np.exp(-d / self.tau_early) - 1.0)
        return float(np.exp(d / self.tau_late) - 1.0)

    def _greedy_match(self, true_ilocs, pred_ilocs):
        """Greedy-match each prediction to its nearest unmatched true event.

        Predictions are processed in ascending time order. For each
        prediction, the nearest yet-unmatched true event (by absolute time)
        is selected. This mirrors the matching style used in other sktime
        detection metrics (e.g. the in-flight ``DetectionDelayMean`` in
        #9895).

        Parameters
        ----------
        true_ilocs : list of int
            Ground truth event positions.
        pred_ilocs : list of int
            Predicted event positions.

        Returns
        -------
        matches : list of (int, int)
            Pairs ``(t_true, t_pred)`` for matched events.
        unmatched_true : list of int
            True events with no matched prediction (missed detections).
        unmatched_pred : list of int
            Predictions with no matched true event (false alarms).
        """
        true_sorted = sorted(true_ilocs)
        pred_sorted = sorted(pred_ilocs)

        used = [False] * len(true_sorted)
        matches = []
        unmatched_pred = []

        for p in pred_sorted:
            best_i = -1
            best_dist = float("inf")
            for i, t in enumerate(true_sorted):
                if used[i]:
                    continue
                dist = abs(p - t)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            if best_i == -1:
                unmatched_pred.append(p)
            else:
                used[best_i] = True
                matches.append((true_sorted[best_i], p))

        unmatched_true = [
            true_sorted[i] for i in range(len(true_sorted)) if not used[i]
        ]
        return matches, unmatched_true, unmatched_pred

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute the PHM asymmetric score on matched event pairs.

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events in "points" format. Must have a column
            ``ilocs`` with integer-like index positions.
        y_pred : pd.DataFrame
            Predicted events in "points" format, same column convention.
        X : pd.DataFrame, optional
            Unused; part of the signature.

        Returns
        -------
        float
            PHM score. ``0.0`` means perfect detection; larger means worse.
        """
        true_ilocs = y_true["ilocs"].values.tolist()
        pred_ilocs = y_pred["ilocs"].values.tolist()

        # Edge cases: empty inputs.
        if len(true_ilocs) == 0 and len(pred_ilocs) == 0:
            return 0.0
        if len(true_ilocs) == 0:
            return float(len(pred_ilocs) * self.unmatched_penalty)
        if len(pred_ilocs) == 0:
            return float(len(true_ilocs) * self.unmatched_penalty)

        matches, unmatched_true, unmatched_pred = self._greedy_match(
            true_ilocs, pred_ilocs
        )

        matched_cost = 0.0
        for t_true, t_pred in matches:
            matched_cost += self._asymmetric_cost(t_pred - t_true)

        n_unmatched = len(unmatched_true) + len(unmatched_pred)
        # Guard: ``n_unmatched * inf`` is ``nan`` when ``n_unmatched == 0``
        # but conceptually is ``0`` — add the unmatched term only when there
        # is at least one unmatched event.
        if n_unmatched > 0:
            total = matched_cost + n_unmatched * self.unmatched_penalty
        else:
            total = matched_cost

        if self.aggregation == "mean":
            denom = len(matches) + n_unmatched
            if denom == 0:
                return 0.0
            return float(total / denom)

        return float(total)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            the ``"default"`` set.

        Returns
        -------
        params : list of dict
            Parameter dictionaries to construct testing instances. The
            default set uses ``unmatched_penalty=100.0`` so test scores
            stay finite under the framework's generic test scenarios.
        """
        param0 = {"unmatched_penalty": 100.0}
        param1 = {
            "tau_early": 20.0,
            "tau_late": 5.0,
            "unmatched_penalty": 50.0,
        }
        param2 = {"unmatched_penalty": 100.0, "aggregation": "mean"}
        return [param0, param1, param2]
