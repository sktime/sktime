# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Seeded binary segmentation algorithm for changepoint detection."""

__author__ = ["Tveten"]
__all__ = ["SeededBinarySegmentation"]

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
)
from sktime.detection.base import BaseDetector

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def make_seeded_intervals(n, min_length, max_length, growth_factor=1.5):
    """Create seeded intervals with exponentially growing length.

    Parameters
    ----------
    n : int
        Number of data points.
    min_length : int
        Minimum interval length.
    max_length : int
        Maximum interval length.
    growth_factor : float
        Multiplicative growth factor.

    Returns
    -------
    starts : np.ndarray
    ends : np.ndarray
    """
    max_length = min(max_length, n)
    if max_length < min_length:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    step_factor = 1 - 1 / growth_factor
    n_lengths = max(
        1, int(np.ceil(np.log(max_length / min_length) / np.log(growth_factor)))
    )
    interval_lens = np.unique(np.round(np.geomspace(min_length, max_length, n_lengths)))

    starts_list = []
    ends_list = []
    for interval_len in interval_lens:
        step = max(1, np.round(step_factor * interval_len))
        n_steps = int(np.ceil((n - interval_len) / step))
        for i in range(n_steps + 1):
            s = int(i * step)
            e = int(min(i * step + interval_len, n))
            starts_list.append(s)
            ends_list.append(e)
        if ends_list and ends_list[-1] - starts_list[-1] < min_length:
            starts_list[-1] = n - min_length

    return np.array(starts_list, dtype=np.int64), np.array(ends_list, dtype=np.int64)


def _greedy_selection(max_scores, argmax_scores, starts, ends):
    """Greedy changepoint selection."""
    max_scores = max_scores.copy()
    cpts = []
    while np.any(max_scores > 0):
        argmax = max_scores.argmax()
        cpt = int(argmax_scores[argmax])
        cpts.append(cpt)
        max_scores[(cpt >= starts) & (cpt < ends)] = 0.0
    cpts.sort()
    return cpts


def _narrowest_selection(max_scores, argmax_scores, starts, ends):
    """Narrowest-over-threshold changepoint selection."""
    above = max_scores > 0
    candidate_starts = starts[above]
    candidate_ends = ends[above]
    candidate_maximizers = argmax_scores[above]

    cpts = []
    while len(candidate_starts) > 0:
        argmin = np.argmin(candidate_ends - candidate_starts)
        cpt = int(candidate_maximizers[argmin])
        cpts.append(cpt)
        keep = ~((cpt >= candidate_starts) & (cpt < candidate_ends))
        candidate_starts = candidate_starts[keep]
        candidate_ends = candidate_ends[keep]
        candidate_maximizers = candidate_maximizers[keep]

    cpts.sort()
    return cpts


def _run_seeded_binseg(
    penalised_score,
    X,
    max_interval_length,
    growth_factor,
    selection_method="greedy",
):
    """Run the seeded binary segmentation algorithm.

    Parameters
    ----------
    penalised_score : PenalisedScore
        Penalised change score.
    X : np.ndarray
        2D data array.
    max_interval_length : int
        Maximum interval length.
    growth_factor : float
        Geometric growth factor.
    selection_method : str
        ``"greedy"`` or ``"narrowest"``.

    Returns
    -------
    cpts : list[int]
    max_scores : np.ndarray
    argmax_scores : np.ndarray
    starts : np.ndarray
    ends : np.ndarray
    """
    n_samples = X.shape[0]
    min_size = penalised_score.min_size

    starts, ends = make_seeded_intervals(
        n_samples,
        2 * min_size,
        max_interval_length,
        growth_factor,
    )

    max_scores = np.zeros(starts.size)
    argmax_scores = np.zeros(starts.size, dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        splits = np.arange(start + min_size, end - min_size + 1)
        intervals = np.column_stack(
            (np.repeat(start, splits.size), splits, np.repeat(end, splits.size))
        )
        scores = penalised_score.evaluate(X, intervals)
        argmax = np.argmax(scores)
        max_scores[i] = scores[argmax, 0]
        argmax_scores[i] = splits[0] + argmax

    if selection_method == "greedy":
        cpts = _greedy_selection(max_scores, argmax_scores, starts, ends)
    else:
        cpts = _narrowest_selection(max_scores, argmax_scores, starts, ends)

    return cpts, max_scores, argmax_scores, starts, ends


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SeededBinarySegmentation(BaseDetector):
    """Seeded binary segmentation algorithm for changepoint detection.

    Tests for changepoints in seeded intervals of exponentially growing
    length [1]_, achieving the same theoretical guarantees as binary
    segmentation but in log-linear time.

    Parameters
    ----------
    change_score : BaseIntervalScorer, optional, default=CUSUM()
        Change score (or cost, which is converted automatically).
    penalty : float, np.ndarray, or None, default=None
        Penalty for declaring a changepoint.
    max_interval_length : int, default=200
        Maximum seeded interval length.
    growth_factor : float, default=1.5
        Growth factor for successive interval lengths.  Must be in (1, 2].
    selection_method : str, default="greedy"
        ``"greedy"`` or ``"narrowest"``.

    References
    ----------
    .. [1] Kovács, S. et al. (2023). Seeded binary segmentation.
       Biometrika, 110(1), 249-256.

    Examples
    --------
    >>> from sktime.detection._seeded_binseg import SeededBinarySegmentation
    >>> import numpy as np
    >>> X = np.concatenate([np.zeros(100), 10*np.ones(100), np.zeros(100)])
    >>> det = SeededBinarySegmentation(penalty=30)
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
        max_interval_length=200,
        growth_factor=1.5,
        selection_method="greedy",
    ):
        self.change_score = change_score
        self.penalty = penalty
        self.max_interval_length = max_interval_length
        self.growth_factor = growth_factor
        self.selection_method = selection_method
        super().__init__()

        _score = CUSUM() if change_score is None else change_score
        check_interval_scorer(
            _score,
            "change_score",
            "SeededBinarySegmentation",
            required_tasks=["cost", "change_score"],
            allow_penalised=True,
        )
        _change_score = to_change_score(_score)

        check_penalty(penalty, "penalty", "SeededBinarySegmentation")
        self._penalised_score = (
            _change_score.clone()
            if _change_score.get_tag("is_penalised")
            else PenalisedScore(_change_score, penalty)
        )

        check_in_interval(
            pd.Interval(1.0, 2.0, closed="right"),
            self.growth_factor,
            "growth_factor",
        )
        valid_selection_methods = ["greedy", "narrowest"]
        if self.selection_method not in valid_selection_methods:
            raise ValueError(
                f"selection_method must be one of {valid_selection_methods}."
            )

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        X_df = check_data(X, min_length=2, min_length_name="2")
        X_arr = as_2d_array(X_df)

        check_larger_than(
            2 * self._penalised_score.min_size,
            self.max_interval_length,
            "max_interval_length",
        )
        cpts, scores, maximizers, starts, ends = _run_seeded_binseg(
            penalised_score=self._penalised_score,
            X=X_arr,
            max_interval_length=self.max_interval_length,
            growth_factor=self.growth_factor,
            selection_method=self.selection_method,
        )
        return format_changepoints(np.array(cpts))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {"change_score": L2Cost(), "max_interval_length": 100, "penalty": 30},
            {"change_score": L2Cost(), "max_interval_length": 20, "penalty": 10},
            {
                "change_score": L2Cost(),
                "max_interval_length": 20,
                "penalty": 10,
                "selection_method": "narrowest",
            },
        ]
