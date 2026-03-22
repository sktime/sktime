# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Penalised interval scorer — stateless wrapper."""

__author__ = ["Tveten"]

import numpy as np

from sktime.detection._penalties import make_bic_penalty
from sktime.detection._utils import (
    check_interval_scorer,
    check_penalty,
    check_penalty_against_data,
)
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _penalise_scores_constant(scores, penalty):
    """Penalise scores with a constant penalty.

    Parameters
    ----------
    scores : np.ndarray
        2D score array (n_cuts, n_variables).
    penalty : float
        Constant penalty value.

    Returns
    -------
    np.ndarray
        1D penalised scores (n_cuts,).
    """
    return scores.sum(axis=1) - penalty


def _penalise_scores_linear(scores, penalty_values):
    """Penalise scores with a linear penalty.

    Parameters
    ----------
    scores : np.ndarray
        2D score array (n_cuts, n_variables).
    penalty_values : np.ndarray
        Linear penalty array.

    Returns
    -------
    np.ndarray
        1D penalised scores (n_cuts,).
    """
    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope
    penalised = np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    return penalised.sum(axis=1)


def _penalise_scores_nonlinear(scores, penalty_values):
    """Penalise scores with a nonlinear penalty.

    Parameters
    ----------
    scores : np.ndarray
        2D score array (n_cuts, n_variables).
    penalty_values : np.ndarray
        Nonlinear penalty array.

    Returns
    -------
    np.ndarray
        1D penalised scores (n_cuts,).
    """
    penalised = np.empty(scores.shape[0])
    for i in range(scores.shape[0]):
        sorted_scores = np.sort(scores[i])[::-1]
        cumulative = np.cumsum(sorted_scores) - penalty_values
        penalised[i] = np.max(cumulative)
    return penalised


class PenalisedScore(BaseIntervalScorer):
    """Penalised interval scorer.

    Wraps an interval scorer: aggregates and penalises scores across variables.

    * Constant penalty: sum of scores minus penalty.
    * Array penalty: optimal penalised score over the number of affected
      components [1]_ [2]_.

    Parameters
    ----------
    score : BaseIntervalScorer
        The score to penalise. Must have ``task`` in
        ``{"change_score", "local_anomaly_score", "saving"}``.
    penalty : float, np.ndarray, or None, default=None
        If ``float``: constant penalty.
        If ``np.ndarray``: element ``i`` is the penalty for ``i+1`` affected
        variables (must be positive and non-decreasing).
        If ``None``: BIC penalty computed from data at evaluation time.
    make_default_penalty : callable or None, default=None
        ``(score, n, p) -> float | np.ndarray``. Used when ``penalty=None``.
        Default uses ``make_bic_penalty(score.get_model_size(p), n)``.

    References
    ----------
    .. [1] Fisch et al. (2022). Subset multivariate segment and point anomaly
       detection.
    .. [2] Tickle et al. (2021). A computationally efficient, high-dimensional
       multiple changepoint procedure.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "is_aggregated": True,
        "is_penalised": True,
    }

    def __init__(self, score, penalty=None, make_default_penalty=None):
        self.score = score
        self.penalty = penalty
        self.make_default_penalty = make_default_penalty
        super().__init__()

        check_interval_scorer(
            score,
            "score",
            "PenalisedScore",
            required_tasks=["change_score", "local_anomaly_score", "saving"],
            allow_penalised=False,
        )
        check_penalty(penalty, "penalty", "PenalisedScore")

        if (
            score.get_tag("is_aggregated")
            and isinstance(penalty, np.ndarray)
            and penalty.size > 1
        ):
            raise ValueError(
                "`penalty` must be a constant for aggregated scores. "
                f"Got `score={score.__class__.__name__}`."
            )

        self.clone_tags(score, ["task", "distribution_type", "is_conditional"])

    @property
    def min_size(self):
        return self.score.min_size

    def get_model_size(self, p):
        return self.score.get_model_size(p)

    def _evaluate(self, X, cuts):
        scores = self.score.evaluate(X, cuts)
        n, p = X.shape

        if self.penalty is not None:
            penalty = self.penalty
        elif self.make_default_penalty is not None:
            penalty = self.make_default_penalty(self.score, n, p)
        else:
            penalty = make_bic_penalty(self.score.get_model_size(p), n)

        check_penalty_against_data(penalty, X, "PenalisedScore")

        penalty_arr = np.asarray(penalty).flatten()
        penalty_diff = np.diff(penalty_arr)

        if p == 1 or penalty_arr.size == 1:
            penalise = _penalise_scores_constant
        elif np.allclose(penalty_diff, penalty_diff[0]):
            penalise = _penalise_scores_linear
        else:
            penalise = _penalise_scores_nonlinear

        return penalise(scores, penalty).reshape(-1, 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._anomaly_scores._l2_saving import L2Saving
        from sktime.detection._change_scores._multivariate_gaussian_score import (
            MultivariateGaussianScore,
        )

        return [
            {"score": L2Saving(), "penalty": 20},
            {"score": MultivariateGaussianScore()},
        ]
