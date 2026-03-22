# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Collective and point anomaly (CAPA) detection algorithm."""

__author__ = ["Tveten"]
__all__ = ["CAPA"]

import numpy as np

from sktime.detection._anomaly_scores._from_cost import to_saving
from sktime.detection._anomaly_scores._l2_saving import L2Saving
from sktime.detection._compose import PenalisedScore
from sktime.detection._formatters import format_anomaly_points
from sktime.detection._penalties import (
    make_bic_penalty,
    make_chi2_penalty,
    make_linear_chi2_penalty,
)
from sktime.detection._utils import (
    as_2d_array,
    check_data,
    check_interval_scorer,
    check_larger_than,
    check_penalty,
    check_smaller_than,
)
from sktime.detection.base import BaseDetector

# ---------------------------------------------------------------------------
# Penalty helpers  (callbacks for PenalisedScore.make_default_penalty)
# ---------------------------------------------------------------------------


def _make_chi2_penalty(score, n, p):
    """Create a chi-square penalty from scorer and data dimensions."""
    return make_chi2_penalty(score.get_model_size(p), n)


def _make_linear_chi2_penalty(score, n, p):
    """Create a linear chi-square penalty from scorer and data dimensions."""
    return make_linear_chi2_penalty(score.get_model_size(p), n, p)


def _resolve_penalty(penalised_score, n, p):
    """Resolve the effective penalty value for a PenalisedScore.

    Replicates the logic inside ``PenalisedScore._evaluate`` so that
    the penalty can be obtained without running a full evaluation.
    """
    if penalised_score.penalty is not None:
        return penalised_score.penalty
    elif penalised_score.make_default_penalty is not None:
        return penalised_score.make_default_penalty(penalised_score.score, n, p)
    else:
        return make_bic_penalty(penalised_score.score.get_model_size(p), n)


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------


def _get_anomalies(anomaly_starts):
    """Backtrack optimal anomaly starts to recover anomalies.

    Parameters
    ----------
    anomaly_starts : np.ndarray
        Array where ``anomaly_starts[i]`` is the optimal anomaly start
        ending at position ``i``, or ``NaN`` if no anomaly ends there.

    Returns
    -------
    segment_anomalies : list of (int, int)
    point_anomalies : list of (int, int)
    """
    segment_anomalies = []
    point_anomalies = []
    i = anomaly_starts.size - 1
    while i >= 0:
        if np.isnan(anomaly_starts[i]):
            i -= 1
            continue
        start_i = int(anomaly_starts[i])
        size = i - start_i + 1
        if size > 1:
            segment_anomalies.append((start_i, i + 1))
            i = start_i
        elif size == 1:
            point_anomalies.append((i, i + 1))
        i -= 1
    return segment_anomalies, point_anomalies


def _get_affected_components(score, X, anomalies, penalty):
    """Identify which variables are affected for each anomaly.

    Parameters
    ----------
    score : BaseIntervalScorer
        The inner (unpenalised) saving score.
    X : np.ndarray
        Data array of shape ``(n, p)``.
    anomalies : list of (int, int)
        Anomaly ``(start, end)`` pairs.
    penalty : float or np.ndarray
        Effective penalty used during detection.

    Returns
    -------
    list of (int, int, np.ndarray)
        Each tuple is ``(start, end, affected_columns)``.
    """
    new_anomalies = []
    for start, end in anomalies:
        interval = np.array([[start, end]])
        saving_values = score.evaluate(X, interval)[0]
        saving_order = np.argsort(-saving_values)  # Decreasing order.
        penalised_savings = np.cumsum(saving_values[saving_order]) - penalty
        argmax = np.argmax(penalised_savings)
        new_anomalies.append((start, end, saving_order[: argmax + 1]))
    return new_anomalies


def _run_capa(
    segment_penalised_saving,
    point_penalised_saving,
    X,
    min_segment_length,
    max_segment_length,
    segment_penalty,
):
    """Run the CAPA algorithm.

    Parameters
    ----------
    segment_penalised_saving : PenalisedScore
        Penalised segment saving scorer.
    point_penalised_saving : PenalisedScore
        Penalised point saving scorer.
    X : np.ndarray
        Data array of shape ``(n, p)``.
    min_segment_length : int
        Minimum segment length.
    max_segment_length : int
        Maximum segment length.
    segment_penalty : float or np.ndarray
        Resolved segment penalty (for pruning).

    Returns
    -------
    opt_savings : np.ndarray
        Cumulative optimal savings of length ``n``.
    segment_anomalies : list of (int, int)
    point_anomalies : list of (int, int)
    """
    n_samples = X.shape[0]

    opt_savings = np.zeros(n_samples + 1)
    opt_anomaly_starts = np.repeat(np.nan, n_samples)
    starts = np.array([], dtype=int)
    max_segment_penalty = float(np.max(np.asarray(segment_penalty)))

    ts = np.arange(min_segment_length - 1, n_samples)
    for t in ts:
        t_array = np.array([t])

        # ---- segment anomalies ----
        starts = np.concatenate((starts, t_array - min_segment_length + 1))
        ends = np.repeat(t + 1, len(starts))
        intervals = np.column_stack((starts, ends))
        segment_savings = segment_penalised_saving.evaluate(X, intervals).flatten()
        candidate_savings = opt_savings[starts] + segment_savings
        candidate_argmax = int(np.argmax(candidate_savings))
        opt_segment_saving = candidate_savings[candidate_argmax]
        opt_start = starts[0] + candidate_argmax

        # ---- point anomalies ----
        point_interval = np.column_stack((t_array, t_array + 1))
        point_savings = point_penalised_saving.evaluate(X, point_interval).flatten()
        opt_point_saving = opt_savings[t] + point_savings[0]

        # ---- combine and store ----
        savings = np.array([opt_savings[t], opt_segment_saving, opt_point_saving])
        argmax = int(np.argmax(savings))
        opt_savings[t + 1] = savings[argmax]
        if argmax == 1:
            opt_anomaly_starts[t] = opt_start
        elif argmax == 2:
            opt_anomaly_starts[t] = t

        # ---- pruning ----
        saving_too_low = candidate_savings + max_segment_penalty <= opt_savings[t + 1]
        too_long_segment = starts < t - max_segment_length + 2
        prune = saving_too_low | too_long_segment
        starts = starts[~prune]

    segment_anomalies, point_anomalies = _get_anomalies(opt_anomaly_starts)
    return opt_savings[1:], segment_anomalies, point_anomalies


def _check_capa_penalised_score(score, name, caller_name):
    """Validate that a pre-penalised score is a ``PenalisedScore``."""
    if score.get_tag("is_penalised") and not isinstance(score, PenalisedScore):
        raise ValueError(
            f"{caller_name} only supports a penalised `{name}` constructed"
            " by `PenalisedScore`."
        )


def _make_nonlinear_chi2_penalty(score, n, p):
    """Nonlinear chi-square penalty callback for CAPA testing."""
    from sktime.detection._penalties import make_nonlinear_chi2_penalty

    return make_nonlinear_chi2_penalty(score.get_model_size(p), n, p)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class CAPA(BaseDetector):
    """Collective and point anomaly (CAPA) detection algorithm.

    Efficient implementation of the CAPA family for anomaly detection.
    Supports both univariate data [1]_ (CAPA) and multivariate data
    with subset anomalies [2]_ (MVCAPA) using penalised savings.

    Parameters
    ----------
    segment_saving : BaseIntervalScorer or None, default=None
        Saving for segment anomaly detection.  If ``None``, ``L2Saving()``
        is used.  A cost is automatically converted via ``to_saving``.
        If pre-penalised, must be a ``PenalisedScore``.
    segment_penalty : float, np.ndarray or None, default=None
        Penalty for segment anomalies.  Ignored when ``segment_saving``
        is already penalised.  ``None`` creates a chi-square penalty.
    point_saving : BaseIntervalScorer or None, default=None
        Saving for point anomaly detection (must have ``min_size == 1``).
        Defaults to ``L2Saving()`` when ``None``.
    point_penalty : float, np.ndarray or None, default=None
        Penalty for point anomalies.  ``None`` creates a linear
        chi-square penalty.
    min_segment_length : int, default=2
        Minimum anomaly segment length.
    max_segment_length : int, default=1000
        Maximum anomaly segment length.
    ignore_point_anomalies : bool, default=False
        If ``True``, only segment anomalies are returned.
    find_affected_components : bool, default=False
        If ``True``, return affected variable indices per anomaly.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022).
       A linear time method for the detection of collective and point
       anomalies.
    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022).
       Subset multivariate collective and point anomaly detection.
    .. [3] Tveten, M., Eckley, I. A., & Fearnhead, P. (2022).
       Scalable change-point and anomaly detection in cross-correlated
       data with an application to condition monitoring.

    Examples
    --------
    >>> from sktime.detection._capa import CAPA
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> X = pd.DataFrame(rng.standard_normal((200, 1)))
    >>> X.iloc[80:100] += 10.0
    >>> detector = CAPA(min_segment_length=5, max_segment_length=100)
    >>> detector.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
    }

    def __init__(
        self,
        segment_saving=None,
        segment_penalty=None,
        point_saving=None,
        point_penalty=None,
        min_segment_length=2,
        max_segment_length=1000,
        ignore_point_anomalies=False,
        find_affected_components=False,
    ):
        self.segment_saving = segment_saving
        self.segment_penalty = segment_penalty
        self.point_saving = point_saving
        self.point_penalty = point_penalty
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        self.find_affected_components = find_affected_components
        super().__init__()

        # ---- segment saving setup ----
        _segment_score = L2Saving() if segment_saving is None else segment_saving
        check_interval_scorer(
            _segment_score,
            "segment_saving",
            "CAPA",
            required_tasks=["cost", "saving"],
            allow_penalised=True,
        )
        _segment_saving = to_saving(_segment_score)
        _check_capa_penalised_score(_segment_saving, "segment_saving", "CAPA")

        check_penalty(segment_penalty, "segment_penalty", "CAPA")
        self._segment_penalised_saving = (
            _segment_saving.clone()
            if _segment_saving.get_tag("is_penalised")
            else PenalisedScore(
                _segment_saving,
                segment_penalty,
                make_default_penalty=_make_chi2_penalty,
            )
        )

        # ---- point saving setup ----
        _point_score = L2Saving() if point_saving is None else point_saving
        check_interval_scorer(
            _point_score,
            "point_saving",
            "CAPA",
            required_tasks=["cost", "saving"],
            allow_penalised=True,
        )
        if _point_score.min_size != 1:
            raise ValueError("`point_saving` must have `min_size == 1`.")
        _point_saving = to_saving(_point_score)
        _check_capa_penalised_score(_point_saving, "point_saving", "CAPA")

        check_penalty(point_penalty, "point_penalty", "CAPA")
        self._point_penalised_saving = (
            _point_saving.clone()
            if _point_saving.get_tag("is_penalised")
            else PenalisedScore(
                _point_saving,
                point_penalty,
                make_default_penalty=_make_linear_chi2_penalty,
            )
        )

        # ---- validate lengths ----
        check_larger_than(2, min_segment_length, "min_segment_length")
        check_larger_than(min_segment_length, max_segment_length, "max_segment_length")

    def _fit(self, X, y=None):
        """No-op (stateless detector)."""
        return self

    def _predict(self, X):
        """Detect collective and point anomalies in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data.

        Returns
        -------
        pd.DataFrame
            ``"ilocs"`` column with integer point indices of anomalous
            time points.
        """
        X_checked = check_data(
            X,
            min_length=self.min_segment_length,
            min_length_name="min_segment_length",
        )
        X_arr = as_2d_array(X_checked)
        n, p = X_arr.shape

        min_seg = max(
            self.min_segment_length,
            self._segment_penalised_saving.min_size,
        )
        check_smaller_than(self.max_segment_length, min_seg, "min_segment_length")
        check_larger_than(min_seg, n, "X.shape[0]")

        # Resolve the segment penalty for pruning inside _run_capa
        segment_penalty = _resolve_penalty(self._segment_penalised_saving, n, p)

        opt_savings, segment_anomalies, point_anomalies = _run_capa(
            segment_penalised_saving=self._segment_penalised_saving,
            point_penalised_saving=self._point_penalised_saving,
            X=X_arr,
            min_segment_length=min_seg,
            max_segment_length=self.max_segment_length,
            segment_penalty=segment_penalty,
        )
        if self.find_affected_components:
            seg_pen = _resolve_penalty(self._segment_penalised_saving, n, p)
            segment_anomalies = _get_affected_components(
                self._segment_penalised_saving.score,
                X_arr,
                segment_anomalies,
                seg_pen,
            )
            pt_pen = _resolve_penalty(self._point_penalised_saving, n, p)
            point_anomalies = _get_affected_components(
                self._point_penalised_saving.score,
                X_arr,
                point_anomalies,
                pt_pen,
            )

        anomalies = segment_anomalies
        if not self.ignore_point_anomalies:
            anomalies = anomalies + point_anomalies
        # Sort by start position (works for both (start,end) and
        # (start,end,cols) tuples).
        anomalies = sorted(anomalies)

        # Extract (start, end) pairs for formatting
        anomaly_pairs = [(a[0], a[1]) for a in anomalies]
        return format_anomaly_points(anomaly_pairs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._anomaly_scores._l2_saving import L2Saving
        from sktime.detection._compose import PenalisedScore

        params = [
            {
                "segment_saving": PenalisedScore(
                    L2Saving(),
                    make_default_penalty=_make_nonlinear_chi2_penalty,
                ),
                "min_segment_length": 5,
                "max_segment_length": 100,
                "find_affected_components": True,
            },
            {
                "segment_saving": L2Saving(),
                "segment_penalty": 30.0,
                "min_segment_length": 5,
                "max_segment_length": 100,
            },
        ]
        return params
