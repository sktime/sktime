# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Circular binary segmentation algorithm for segment anomaly detection."""

__author__ = ["Tveten"]
__all__ = ["CircularBinarySegmentation"]

import numpy as np
import pandas as pd

from sktime.detection._anomaly_scores._from_cost import to_local_anomaly_score
from sktime.detection._compose import PenalisedScore
from sktime.detection._formatters import format_segments
from sktime.detection._penalties import make_bic_penalty
from sktime.detection._utils import (
    as_2d_array,
    check_data,
    check_in_interval,
    check_interval_scorer,
    check_larger_than,
    check_penalty,
)
from sktime.detection.base import BaseDetector
from sktime.detection.costs._l2_cost import L2Cost
from sktime.detection.seeded_binseg import make_seeded_intervals

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _greedy_anomaly_selection(
    penalised_scores,
    anomaly_starts,
    anomaly_ends,
    starts,
    ends,
):
    """Select non-overlapping anomalies greedily by score.

    Parameters
    ----------
    penalised_scores : np.ndarray
        1D array of penalised anomaly scores per seeded interval.
    anomaly_starts : np.ndarray
        1D array of best anomaly start per seeded interval.
    anomaly_ends : np.ndarray
        1D array of best anomaly end per seeded interval.
    starts : np.ndarray
        Seeded interval starts.
    ends : np.ndarray
        Seeded interval ends.

    Returns
    -------
    list of (int, int)
        Non-overlapping anomaly segments sorted by start.
    """
    penalised_scores = penalised_scores.copy()
    anomalies = []
    while np.any(penalised_scores > 0):
        argmax = int(penalised_scores.argmax())
        anomaly_start = int(anomaly_starts[argmax])
        anomaly_end = int(anomaly_ends[argmax])
        anomalies.append((anomaly_start, anomaly_end))
        # Remove intervals overlapping with the detected anomaly.
        overlap = (anomaly_end > starts) & (anomaly_start < ends)
        penalised_scores[overlap] = 0.0
    anomalies.sort()
    return anomalies


def _make_anomaly_intervals(interval_start, interval_end, min_segment_length=1):
    """Generate all candidate anomaly (inner) intervals within an outer interval.

    Parameters
    ----------
    interval_start : int
        Outer interval start.
    interval_end : int
        Outer interval end.
    min_segment_length : int
        Minimum length for inner and surrounding segments.

    Returns
    -------
    starts : np.ndarray
    ends : np.ndarray
    """
    starts_list = []
    ends_list = []
    for i in range(
        interval_start + min_segment_length,
        interval_end - min_segment_length + 2,
    ):
        for j in range(
            i + min_segment_length,
            interval_end - min_segment_length + 1,
        ):
            baseline_n = interval_end - j + i - interval_start
            if baseline_n >= min_segment_length:
                starts_list.append(i)
                ends_list.append(j)
    return np.array(starts_list, dtype=np.int64), np.array(ends_list, dtype=np.int64)


def _make_bic_penalty_cb(score, n, p):
    """BIC penalty callback with two additional changepoint parameters."""
    return make_bic_penalty(score.get_model_size(p), n, additional_cpts=2)


def _run_circular_binseg(
    penalised_score,
    X,
    min_segment_length,
    max_interval_length,
    growth_factor,
):
    """Run circular binary segmentation.

    Parameters
    ----------
    penalised_score : PenalisedScore
        Penalised local anomaly score.
    X : np.ndarray
        Data array of shape ``(n, p)``.
    min_segment_length : int
        Minimum anomaly / baseline segment length.
    max_interval_length : int
        Maximum outer interval length for seeded intervals.
    growth_factor : float
        Seeded interval growth factor.

    Returns
    -------
    anomalies : list of (int, int)
    anomaly_scores : np.ndarray
    starts : np.ndarray
    ends : np.ndarray
    """
    n_samples = X.shape[0]

    starts, ends = make_seeded_intervals(
        n_samples,
        3 * min_segment_length,
        max_interval_length,
        growth_factor,
    )

    n_intervals = starts.size
    anomaly_scores = np.zeros(n_intervals)
    anomaly_starts = np.zeros(n_intervals, dtype=np.int64)
    anomaly_ends = np.zeros(n_intervals, dtype=np.int64)

    for i, (start, end) in enumerate(zip(starts, ends)):
        cand_starts, cand_ends = _make_anomaly_intervals(start, end, min_segment_length)
        if cand_starts.size == 0:
            continue

        intervals = np.column_stack(
            (
                np.repeat(start, cand_starts.size),
                cand_starts,
                cand_ends,
                np.repeat(end, cand_starts.size),
            )
        )
        scores = penalised_score.evaluate(X, intervals)
        agg_scores = np.sum(scores, axis=1)
        argmax = int(np.argmax(agg_scores))
        anomaly_scores[i] = agg_scores[argmax]
        anomaly_starts[i] = cand_starts[argmax]
        anomaly_ends[i] = cand_ends[argmax]

    anomalies = _greedy_anomaly_selection(
        anomaly_scores, anomaly_starts, anomaly_ends, starts, ends
    )
    return anomalies, anomaly_scores, starts, ends


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class CircularBinarySegmentation(BaseDetector):
    """Circular binary segmentation for multiple segment anomaly detection.

    Binary segmentation type changepoint detection algorithms recursively
    split the data into two segments, and test whether the two segments are
    different. Circular binary segmentation [1]_ is a variant of binary
    segmentation where the statistical test (anomaly score) is applied to
    compare the data behaviour of an inner interval subset with the
    surrounding data contained in an outer interval.
    In other words, the null hypothesis within each outer interval is that the
    data is stationary, while the alternative hypothesis is that there is a
    segment anomaly within the outer interval.

    Parameters
    ----------
    anomaly_score : BaseIntervalScorer, optional, default=L2Cost()
        The local anomaly score to use for anomaly detection. If a cost is
        given, it is converted to a local anomaly score using the
        ``LocalAnomalyScore`` class.
    penalty : np.ndarray or float, optional, default=None
        The penalty to use for anomaly detection. If the anomaly score is
        penalised (``anomaly_score.get_tag("is_penalised")``) the penalty will
        be ignored. The different types of penalties are:

        * ``float``: A constant penalty applied to the sum of scores across
          all variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty
          for ``i+1`` variables being affected by an anomaly. The penalty array
          must be positive and increasing (not strictly). A penalised score
          with a linear penalty array is faster to evaluate than a nonlinear
          penalty array.
        * ``None``: A default penalty is created in ``predict`` based on the
          fitted score using the ``make_bic_penalty`` function.

    min_segment_length : int, default=5
        Minimum length between two changepoints. Must be greater than or equal
        to 1.
    max_interval_length : int, default=1000
        The maximum length of an interval to estimate a changepoint in. Must
        be greater than or equal to ``2 * min_segment_length``.
    growth_factor : float, default=1.5
        The growth factor for the seeded intervals. Intervals grow in size
        according to
        ``interval_len = max(interval_len + 1,
        floor(growth_factor * interval_len))``,
        starting at ``interval_len = min_interval_length``. It also governs
        the amount of overlap between intervals of the same length, as the
        start of each interval is shifted by a factor of
        ``1 + 1 / growth_factor``. Must be a float in ``(1, 2]``.

    References
    ----------
    .. [1] Olshen, A. B., Venkatraman, E. S., Lucito, R., & Wigler, M.
        (2004). Circular binary segmentation for the analysis of array-based
        DNA copy number data. Biostatistics, 5(4), 557-572.

    Examples
    --------
    >>> from sktime.detection.circular_binseg import CircularBinarySegmentation
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> X = pd.DataFrame(rng.standard_normal((75, 1)))
    >>> X.iloc[20:30] += 10.0
    >>> X.iloc[50:55] += 20.0
    >>> detector = CircularBinarySegmentation(penalty=20.0)
    >>> detector.fit_predict(X)  # doctest: +SKIP

    Notes
    -----
    Using costs to generate local anomaly scores will be significantly slower
    than using anomaly scores that are implemented directly. This is because
    the local anomaly score requires evaluating the cost at disjoint subsets
    of the data (before and after an anomaly), which is not a natural
    operation for costs implemented as interval evaluators.
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(
        self,
        anomaly_score=None,
        penalty=None,
        min_segment_length=5,
        max_interval_length=1000,
        growth_factor=1.5,
    ):
        self.anomaly_score = anomaly_score
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        super().__init__()

        _score = L2Cost() if anomaly_score is None else anomaly_score
        check_interval_scorer(
            _score,
            "anomaly_score",
            "CircularBinarySegmentation",
            required_tasks=["cost", "local_anomaly_score"],
            allow_penalised=True,
        )
        _anomaly_score = to_local_anomaly_score(_score)

        check_penalty(penalty, "penalty", "CircularBinarySegmentation")
        self._penalised_score = (
            _anomaly_score.clone()
            if _anomaly_score.get_tag("is_penalised")
            else PenalisedScore(
                _anomaly_score,
                penalty,
                make_default_penalty=_make_bic_penalty_cb,
            )
        )

        check_larger_than(1.0, self.min_segment_length, "min_segment_length")
        check_larger_than(
            2 * self.min_segment_length,
            self.max_interval_length,
            "max_interval_length",
        )
        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
        )

    def _fit(self, X, y=None):
        """No-op (stateless detector)."""
        return self

    def _predict(self, X):
        """Detect segment anomalies in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data.

        Returns
        -------
        pd.DataFrame
            ``"ilocs"`` column with ``pd.Interval`` objects representing
            anomalous segments (left-closed).
        """
        X_checked = check_data(
            X,
            min_length=2 * self._penalised_score.min_size,
            min_length_name="2 * anomaly_score.min_size",
        )
        X_arr = as_2d_array(X_checked)

        anomalies, scores, starts, ends = _run_circular_binseg(
            penalised_score=self._penalised_score,
            X=X_arr,
            min_segment_length=self.min_segment_length,
            max_interval_length=self.max_interval_length,
            growth_factor=self.growth_factor,
        )

        if not anomalies:
            return format_segments(np.array([]), np.array([]))
        anom_starts = np.array([a[0] for a in anomalies])
        anom_ends = np.array([a[1] for a in anomalies])
        return format_segments(anom_starts, anom_ends)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection.costs._l2_cost import L2Cost

        params = [
            {"anomaly_score": L2Cost(), "penalty": 20},
            {
                "anomaly_score": L2Cost(),
                "min_segment_length": 3,
                "max_interval_length": 50,
            },
        ]
        return params
