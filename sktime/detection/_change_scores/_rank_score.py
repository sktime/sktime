# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Rank-based change score for multivariate data."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._rank_cost import _compute_ranks_and_pinv_cdf_cov
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _compute_sorted_ranks(centered_data_ranks, segment_start, segment_end, output):
    """Compute sorted ranks for a given segment.

    Parameters
    ----------
    centered_data_ranks : np.ndarray
        Centered data ranks for the full dataset.
    segment_start : int
        Start index (inclusive).
    segment_end : int
        End index (exclusive).
    output : np.ndarray
        Output array to store computed segment ranks.
    """
    n_vars = centered_data_ranks.shape[1]
    seg = centered_data_ranks[segment_start:segment_end, :]
    sorted_seg = np.sort(seg, axis=0)
    n = segment_end - segment_start

    for col in range(n_vars):
        upper = 1 + np.searchsorted(sorted_seg[:, col], seg[:, col], side="left")
        lower = np.searchsorted(sorted_seg[:, col], seg[:, col], side="right")
        output[:n, col] = (upper + lower) / 2.0

    output[:n, :] -= (n + 1) / 2.0


def _direct_rank_score(change_cuts, centered_data_ranks, pinv_rank_cov):
    """Compute the rank-based change score for segments.

    Parameters
    ----------
    change_cuts : np.ndarray
        Array of shape (n_cuts, 3) with [start, split, end].
    centered_data_ranks : np.ndarray
        Centered data ranks for the full dataset.
    pinv_rank_cov : np.ndarray
        Pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    np.ndarray
        1D array of rank-based change scores.
    """
    n_vars = centered_data_ranks.shape[1]
    rank_scores = np.zeros(change_cuts.shape[0])
    if len(rank_scores) == 0:
        return rank_scores

    mean_segment_ranks = np.zeros(n_vars)
    max_len = int(np.max(np.diff(change_cuts[:, [0, 2]], axis=1)))
    segment_data_ranks = np.zeros((max_len, n_vars))

    prev_start = change_cuts[0, 0]
    prev_end = change_cuts[0, 2]
    _compute_sorted_ranks(centered_data_ranks, prev_start, prev_end, segment_data_ranks)

    for i in range(change_cuts.shape[0]):
        seg_start, seg_split, seg_end = change_cuts[i]
        full_len = seg_end - seg_start
        pre_len = seg_split - seg_start
        post_len = seg_end - seg_split

        norm = 2.0 / np.sqrt(full_len * pre_len * post_len)

        if seg_start != prev_start or seg_end != prev_end:
            _compute_sorted_ranks(
                centered_data_ranks, seg_start, seg_end, segment_data_ranks
            )
            prev_start = seg_start
            prev_end = seg_end

        if pre_len < post_len:
            mean_segment_ranks[:] = (
                -np.sum(segment_data_ranks[:pre_len, :], axis=0) * norm
            )
        else:
            mean_segment_ranks[:] = (
                np.sum(segment_data_ranks[:pre_len, :], axis=0) * norm
            )

        rank_scores[i] = mean_segment_ranks @ pinv_rank_cov @ mean_segment_ranks

    return rank_scores


class RankScore(BaseIntervalScorer):
    """Rank-based change score for multivariate data.

    Uses mean rank statistics to detect changes in the distribution of
    multivariate data. The score measures the difference in mean ranks for
    each variable before and after a split, normalized by the pseudo-inverse
    of the rank covariance matrix [1]_.

    Requires sorting the data within each segment, so computational cost per
    evaluation is higher than cumulative-sum-based scores.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Levy-Leduc, C., & Cappe, O. (2015).
       Homogeneity and change-point detection tests for multivariate data using
       rank statistics.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
        "is_aggregated": True,
    }

    def __init__(self):
        super().__init__()

    @property
    def min_size(self):
        return 2

    def get_model_size(self, p):
        return p * (p + 1) // 2

    def _evaluate(self, X, cuts):
        centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)
        scores = _direct_rank_score(cuts, centered_data_ranks, pinv_rank_cov)
        return scores.reshape(-1, 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}, {}]
