"""Independent t-test evaluator for benchmark results."""

__all__ = ["TwoSampleTTest"]

import itertools

import numpy as np
import pandas as pd

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer


class TwoSampleTTest(BaseBenchmarkAnalyzer):
    """Independent two-sample t-test on each ordered pair of estimators.

    Reproduces the legacy ``Evaluator.t_test`` (``scipy.stats.ttest_ind``) and,
    via ``correction="bonferroni"``, the legacy
    ``Evaluator.t_test_with_bonferroni_correction``: a ``"significant"`` column
    is added that flags pairs whose p-value falls below the Bonferroni-corrected
    threshold ``alpha / n_pairs``.

    Parameters
    ----------
    metric : str, optional (default=None)
        Metric to analyse; see ``BaseBenchmarkAnalyzer``.
    lower_is_better : bool, optional (default=True)
        Direction of the metric; not used by the t-test itself but kept for a
        consistent evaluator interface.
    correction : str, optional (default="none")
        Multiple-comparison correction. One of ``"none"`` or ``"bonferroni"``.
    alpha : float, optional (default=0.05)
        Family-wise significance level, used only when
        ``correction="bonferroni"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``["estimator_1", "estimator_2", "statistic", "p_val"]``, plus a
        boolean ``"significant"`` column when ``correction="bonferroni"``.
    """

    _tags = {"property:analyzer_type": "pairwise"}

    def __init__(
        self, metric=None, lower_is_better=True, correction="none", alpha=0.05
    ):
        self.correction = correction
        self.alpha = alpha
        super().__init__(metric=metric, lower_is_better=lower_is_better)

    def _evaluate(self, scores):
        from scipy.stats import ttest_ind

        if self.correction not in ("none", "bonferroni"):
            raise ValueError(
                "`correction` must be one of 'none' or 'bonferroni', but found: "
                f"{self.correction!r}."
            )

        estimator_scores = self._as_estimator_dict(scores)
        rows = []
        for estim_1, estim_2 in itertools.product(estimator_scores.keys(), repeat=2):
            statistic, p_val = ttest_ind(
                np.asarray(estimator_scores[estim_1]),
                np.asarray(estimator_scores[estim_2]),
            )
            rows.append(
                {
                    "estimator_1": estim_1,
                    "estimator_2": estim_2,
                    "statistic": statistic,
                    "p_val": p_val,
                }
            )
        df = pd.DataFrame(
            rows, columns=["estimator_1", "estimator_2", "statistic", "p_val"]
        )

        if self.correction == "bonferroni":
            n_pairs = len(estimator_scores) ** 2
            df["significant"] = df["p_val"] <= self.alpha / n_pairs

        return df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the evaluator."""
        return [{}, {"correction": "bonferroni", "alpha": 0.05}]
