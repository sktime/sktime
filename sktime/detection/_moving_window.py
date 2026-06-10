# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Moving window (MOSUM) changepoint detection algorithm."""

__author__ = ["Tveten"]
__all__ = ["MovingWindow"]

import numpy as np
import pandas as pd

from sktime.detection._change_scores._cusum import CUSUM
from sktime.detection._change_scores._from_cost import to_change_score
from sktime.detection._compose import PenalisedScore
from sktime.detection._formatters import format_changepoints
from sktime.detection._utils import (
    as_2d_array,
    check_data,
    check_in_interval,
    check_interval_scorer,
    check_larger_than,
    check_penalty,
    where_positive,
)
from sktime.detection.base import BaseDetector

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_extended_moving_window_cuts(n_samples, bandwidth, min_size):
    """Create moving-window cut arrays with CUSUM-boundary extension."""
    splits = np.arange(min_size, n_samples - min_size + 1)

    starts = splits - bandwidth
    starts[starts < 0] = 0
    max_start = n_samples - 2 * bandwidth
    starts[starts > max_start] = max_start

    ends = splits + bandwidth
    ends[ends > n_samples] = n_samples
    min_end = 2 * bandwidth
    ends[ends < min_end] = min_end

    return np.column_stack((starts, splits, ends))


def _transform_moving_window(penalised_score, X, bandwidth):
    """Compute penalised scores across moving windows.

    Returns a 1D array of length ``n_samples`` with ``NaN`` at positions
    that fall outside the window range.
    """
    n_samples = X.shape[0]
    cuts = _make_extended_moving_window_cuts(
        n_samples, bandwidth, penalised_score.min_size
    )
    scores = np.repeat(np.nan, n_samples)
    scores[cuts[:, 1]] = penalised_score.evaluate(X, cuts).reshape(-1)
    return scores


def _transform_multiple_moving_window(penalised_score, X, bandwidths):
    n_samples = X.shape[0]
    scores = np.full((n_samples, len(bandwidths)), np.nan)
    for i, bw in enumerate(bandwidths):
        scores[:, i] = _transform_moving_window(penalised_score, X, bw)
    return scores


def _get_candidate_changepoints(scores):
    """Find candidate changepoints from 1D score array."""
    detection_intervals = where_positive(scores)
    changepoints = []
    for start, end in detection_intervals:
        cpt = start + int(np.argmax(scores[start:end]))
        changepoints.append(cpt)
    return changepoints, detection_intervals


def _select_by_detection_length(scores, min_detection_interval):
    candidate_cpts, detection_intervals = _get_candidate_changepoints(scores)
    return [
        cpt
        for cpt, interval in zip(candidate_cpts, detection_intervals)
        if interval[1] - interval[0] >= min_detection_interval
    ]


def _select_by_local_optimum(scores, selection_bandwidth):
    candidate_cpts, _ = _get_candidate_changepoints(scores)
    return [
        cpt
        for cpt in candidate_cpts
        if np.isclose(
            scores[cpt],
            np.max(
                scores[
                    max(cpt - selection_bandwidth, 0) : cpt + selection_bandwidth + 1
                ]
            ),
        )
    ]


def _select_by_bottom_up(scores, bandwidths, local_optimum_fraction):
    bandwidths_sorted = sorted(bandwidths)
    candidate_cpts = []
    for i, bw in enumerate(bandwidths_sorted):
        lo_bw = int(local_optimum_fraction * bw)
        cpts_bw = _select_by_local_optimum(scores[:, i], lo_bw)
        for cpt in cpts_bw:
            candidate_cpts.append((cpt, bw))

    if not candidate_cpts:
        return []

    cpts = [candidate_cpts[0][0]]
    for candidate_cpt, bw in candidate_cpts[1:]:
        distance_to_closest = np.min(np.abs(candidate_cpt - np.array(cpts)))
        lo_bw = int(local_optimum_fraction * bw)
        if distance_to_closest >= lo_bw:
            cpts.append(candidate_cpt)
    return cpts


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MovingWindow(BaseDetector):
    """Moving window (MOSUM) changepoint detection algorithm.

    Runs a change-score test statistic across the data in a moving-window
    fashion [1]_, generalised to arbitrary penalised/unpenalised change
    scores.  Supports multiple bandwidths with bottom-up merging [2]_.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=CUSUM()
        Change score (or cost, which is converted automatically).
    penalty : float, np.ndarray, or None, default=None
        Penalty value.
    bandwidth : int or list of int, default=20
        Window half-width(s).
    selection_method : str, default="local_optimum"
        ``"local_optimum"`` or ``"detection_length"``.
    min_detection_fraction : float, default=0.2
        Minimum detection interval fraction for ``"detection_length"``.
    local_optimum_fraction : float, default=0.4
        Neighbourhood fraction for ``"local_optimum"``.

    References
    ----------
    .. [1] Eichinger, B. & Kirch, C. (2018). A MOSUM procedure for the
       estimation of multiple random change points.
    .. [2] Meier, A., Kirch, C. & Cho, H. (2021). mosum: A package for
       moving sums in change-point analysis. JSS, 97, 1-42.

    Examples
    --------
    >>> from sktime.detection._moving_window import MovingWindow
    >>> import numpy as np
    >>> X = np.concatenate([np.zeros(100), 10*np.ones(100), np.zeros(100)])
    >>> det = MovingWindow(bandwidth=20, penalty=20)
    >>> det.fit(X).predict(X)  # doctest: +SKIP
    """

    _tags = {
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": True,
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
    }

    def __init__(
        self,
        change_score=None,
        penalty=None,
        bandwidth=20,
        selection_method="local_optimum",
        min_detection_fraction=0.2,
        local_optimum_fraction=0.4,
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.bandwidth = bandwidth
        self.selection_method = selection_method
        self.min_detection_fraction = min_detection_fraction
        self.local_optimum_fraction = local_optimum_fraction
        super().__init__()

        _score = CUSUM() if change_score is None else change_score
        check_interval_scorer(
            _score,
            "change_score",
            "MovingWindow",
            required_tasks=["cost", "change_score"],
            allow_penalised=True,
        )
        _change_score = to_change_score(_score)

        check_penalty(penalty, "penalty", "MovingWindow")
        self._penalised_score = (
            _change_score.clone()
            if _change_score.get_tag("is_penalised")
            else PenalisedScore(_change_score, penalty)
        )

        if isinstance(bandwidth, int):
            check_larger_than(1, bandwidth, "bandwidth")
            self._bandwidth = np.array([bandwidth])
        elif isinstance(bandwidth, list):
            if len(bandwidth) == 0:
                raise ValueError("`bandwidth` must be a non-empty list.")
            if not all(isinstance(bw, int) for bw in bandwidth):
                raise TypeError("All elements of `bandwidth` must be integers.")
            if any(bw < 1 for bw in bandwidth):
                raise ValueError("All elements of `bandwidth` must be >= 1.")
            self._bandwidth = np.array(bandwidth)
        else:
            raise TypeError(
                f"`bandwidth` must be int or list of int. Got {type(bandwidth)}."
            )

        check_in_interval(
            pd.Interval(0, 1 / 2, closed="neither"),
            min_detection_fraction,
            "min_detection_fraction",
        )
        check_larger_than(0, local_optimum_fraction, "local_optimum_fraction")

        valid = ["local_optimum", "detection_length"]
        if selection_method not in valid:
            raise ValueError(f"`selection_method` must be one of {valid}.")
        if len(self._bandwidth) > 1 and self.selection_method == "detection_length":
            raise ValueError(
                "'detection_length' is not supported for multiple bandwidths."
            )

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        X_df = check_data(
            X,
            min_length=2 * max(self._bandwidth),
            min_length_name="2*max(bandwidth)",
        )
        X_arr = as_2d_array(X_df)

        scores = _transform_multiple_moving_window(
            self._penalised_score, X_arr, self._bandwidth
        )
        scores_arr = scores

        if self.selection_method == "detection_length":
            min_det_len = int(self.min_detection_fraction * self.bandwidth)
            changepoints = _select_by_detection_length(
                scores_arr.reshape(-1), min_det_len
            )
        elif self.selection_method == "local_optimum" and len(self._bandwidth) == 1:
            lo_bw = int(self.local_optimum_fraction * self.bandwidth)
            changepoints = _select_by_local_optimum(scores_arr.reshape(-1), lo_bw)
        else:
            changepoints = _select_by_bottom_up(
                scores_arr, self._bandwidth, self.local_optimum_fraction
            )

        return format_changepoints(np.array(changepoints))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._gaussian_cost import GaussianCost
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {"change_score": L2Cost(), "bandwidth": 5, "penalty": 20},
            {"change_score": GaussianCost(), "bandwidth": 5, "penalty": 30},
        ]
