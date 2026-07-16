"""Wilcoxon signed-rank test evaluator for benchmark results."""

__all__ = ["WilcoxonEvaluator"]

import itertools

import pandas as pd

from sktime.benchmarking.post_hoc._base import BasePostHocEvaluator


class WilcoxonEvaluator(BasePostHocEvaluator):
    """Wilcoxon signed-rank test on each unique pair of estimators.

    Non-parametric paired test of whether the per-dataset score differences
    between two estimators are symmetric about zero, reproducing the legacy
    ``Evaluator.wilcoxon_test`` (``scipy.stats.wilcoxon``). Computed on each
    unique unordered pair of estimators.

    Returns
    -------
    pandas.DataFrame
        Columns ``["estimator_1", "estimator_2", "statistic", "p_val"]``.
    """

    _tags = {"property:evaluator_type": "pairwise"}

    def _evaluate(self, scores):
        from scipy.stats import wilcoxon

        estimator_scores = self._as_estimator_dict(scores)
        rows = []
        for estim_1, estim_2 in itertools.combinations(estimator_scores.keys(), 2):
            statistic, p_val = wilcoxon(
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
