"""Penalised interval scorer."""

import copy
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..penalties import make_bic_penalty
from ..utils.numba import njit
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.penalties import check_penalty, check_penalty_against_data


@njit
def _penalise_scores_constant(scores: np.ndarray, penalty: float) -> np.ndarray:
    """Penalise scores with a constant penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty: float
        The penalty value.

    Returns
    -------
    penalised_scores : np.ndarray
        The penalised scores.
    """
    penalised_scores = scores.sum(axis=1) - penalty
    return penalised_scores


@njit
def _penalise_scores_linear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with a linear penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty_values : np.ndarray
        The penalty values.

    Returns
    -------
    penalised_savings : np.ndarray
        The penalised savings
    """
    penalty_slope = penalty_values[1] - penalty_values[0]
    penalty_intercept = penalty_values[0] - penalty_slope

    penalised_scores_matrix = (
        np.maximum(scores - penalty_slope, 0.0) - penalty_intercept
    )
    penalised_savings = penalised_scores_matrix.sum(axis=1)
    return penalised_savings


@njit
def _penalise_scores_nonlinear(
    scores: np.ndarray, penalty_values: np.ndarray
) -> np.ndarray:
    """Penalise scores with a nonlinear penalty.

    Parameters
    ----------
    scores : np.ndarray
        The scores to penalise. The output of a BaseIntervalScorer.
    penalty_values : np.ndarray
        The penalty values.

    Returns
    -------
    penalised_scores : np.ndarray
        The penalised scores
    """
    penalised_scores = []
    for score in scores:
        sorted_scores = np.sort(score)[::-1]
        penalised_score = np.cumsum(sorted_scores) - penalty_values
        optimal_penalised_score = np.max(penalised_score)
        penalised_scores.append(optimal_penalised_score)
    return np.array(penalised_scores, dtype=np.float64)


def _make_bic_penalty_from_score(score: BaseIntervalScorer) -> float:
    score.check_is_fitted()
    n = score.n_samples
    p = score.n_variables
    return make_bic_penalty(score.get_model_size(p), n)


class PenalisedScore(BaseIntervalScorer):
    """Penalised interval scorer.

    A wrapper for interval scorers that aggregates and penalises the scores according to
    a penalty function over the set of affected variables as described in [1]_ and [2]_.
    Depending on the type of penalty input, the penalised score is calculated as
    follows:

    * A constant penalty: The penalised score is simply the sum of the scores across
      all variables in the data minus the penalty.
    * A penalty array where element ``i`` of the array is the penalty for ``i+1``
      variables being affected by a change or anomaly: The penalised score is the optimal
      penalised score over the number of affected components. This is suitable for data
      where it is unknown how many variables are affected by a change or anomaly, and to
      obtain strong detection power against both sparse and dense changes or anomalies.

    Parameters
    ----------
    score : BaseIntervalScorer
        The score to penalise. Costs are currently not supported.
    penalty : np.ndarray | float, optional, default=None
        The penalty to use for the penalised score. There are three options:

        * ``float``: A constant penalty applied to the sum of scores across all
          variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty for
          ``i+1`` variables being affected by a change or anomaly. The penalty array
          must be positive and increasing (not strictly). A penalised score with a
          linear penalty array is faster to evaluate than a nonlinear penalty array.
        * ``None``: A default penalty is created in `fit` based on the fitted
          `score` using the `make_default_penalty` function.

    make_default_penalty : Callable, optional, default=None
        A function to create a default penalty from the fitted `score`. The function
        must take a fitted `BaseIntervalScorer` and return a penalty value or
        array. If `None`, the default penalty is created using
        ``make_bic_penalty(score.get_model_size(score.n_variables), score.n_samples)``.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       segment and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    .. [2] Tickle, S. O., Eckley, I. A., & Fearnhead, P. (2021). A computationally
       efficient, high-dimensional multiple changepoint procedure with application to
       global terrorism incidence. Journal of the Royal Statistical Society Series A:
       Statistics in Society, 184(4), 1303-1325.
    """  # noqa: E501

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "is_aggregated": True,
        "is_penalised": True,
    }

    def __init__(
        self,
        score: BaseIntervalScorer,
        penalty: np.ndarray | float | None = None,
        make_default_penalty: Optional[Callable] = None,
    ):
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
                "`penalty` must be a constant penalty (a single value) for aggregated"
                f" scores. Got `score={score.__class__.__name__}`."
            )

        self._make_default_penalty = (
            _make_bic_penalty_from_score
            if make_default_penalty is None
            else make_default_penalty
        )

        self.clone_tags(score, ["task", "distribution_type", "is_conditional"])

    def _fit(self, X: np.ndarray, y=None) -> "PenalisedScore":
        """Fit the penalised interval scorer to training data.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self.score_: BaseIntervalScorer = self.score.clone()
        # Some scores operate on named columns of X, so the columns must be passed on
        # to the internal scorer.
        X_inner = pd.DataFrame(X, columns=self._X_columns, copy=False)
        self.score_.fit(X_inner)

        if self.penalty is None:
            self.penalty_ = self._make_default_penalty(self.score_)
            check_penalty(self.penalty_, "penalty_", "PenalisedScore", allow_none=False)
        else:
            self.penalty_ = copy.deepcopy(self.penalty)
            check_penalty_against_data(self.penalty_, X, "PenalisedScore")

        penalty_array = np.asarray(self.penalty_).flatten()
        penalty_diff = np.diff(penalty_array)
        if X.shape[1] == 1 or penalty_array.size == 1:
            self.penalise_scores = _penalise_scores_constant
        elif np.allclose(penalty_diff, penalty_diff[0]):
            self.penalise_scores = _penalise_scores_linear
        else:
            self.penalise_scores = _penalise_scores_nonlinear

        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the penalised scores according to a set of cuts.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        values : np.ndarray
            A 2D array of scores. One row for each row in cuts.
        """
        scores = self.score_.evaluate(cuts)
        return self.penalise_scores(scores, self.penalty_).reshape(-1, 1)

    @property
    def min_size(self) -> int | None:
        """Minimum valid size of an interval to evaluate.

        The size of each interval is by default defined as ``np.diff(cuts[i, ])``.
        Subclasses can override the min_size to mean something else, for example in
        cases where intervals are combined before evaluation or `cuts` specify
        disjoint intervals.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        if self.is_fitted:
            return self.score_.min_size
        else:
            return self.score.min_size

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters to estimate over each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors. For example, a scorer for a change in mean has one
        parameter to estimate per variable in the data, a scorer for a change in the
        mean and variance has two parameters to estimate per variable, and so on.
        Subclasses should override this method accordingly.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return self.score.get_model_size(p)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.detection._skchange.anomaly_scores import L2Saving
        from sktime.detection._skchange.change_scores import MultivariateGaussianScore

        params = [
            {"score": L2Saving(), "penalty": 20},
            {"score": MultivariateGaussianScore()},
        ]

        return params
