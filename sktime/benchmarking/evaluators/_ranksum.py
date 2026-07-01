"""Wilcoxon rank-sum test evaluator for benchmark results."""

__all__ = ["RanksumEvaluator"]

import itertools

import pandas as pd

from sktime.benchmarking.evaluators._base import BasePostHocEvaluator


class RanksumEvaluator(BasePostHocEvaluator):
    """Wilcoxon rank-sum test on each ordered pair of estimators.

    Non-parametric test of whether two estimators' per-dataset scores are drawn
    from the same distribution, reproducing the legacy
    ``Evaluator.ranksum_test`` (``scipy.stats.ranksums``). Computed for all
    ordered estimator pairs.

    Returns
    -------
    pandas.DataFrame
        Columns ``["estimator_1", "estimator_2", "statistic", "p_val"]``.
    """

    _tags = {"capability:pairwise": True}

    def _evaluate(self, scores):
        from scipy.stats import ranksums

        estimator_scores = self._as_estimator_dict(scores)
        rows = []
        for estim_1, estim_2 in itertools.product(estimator_scores.keys(), repeat=2):
            statistic, p_val = ranksums(
                estimator_scores[estim_1], estimator_scores[estim_2]
            )
            rows.append(
                {
                    "estimator_1": estim_1,
                    "estimator_2": estim_2,
                    "statistic": statistic,
                    "p_val": p_val,
                }
            )
        return pd.DataFrame(
            rows, columns=["estimator_1", "estimator_2", "statistic", "p_val"]
        )
