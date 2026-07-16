"""Sign test evaluator for benchmark results."""

__all__ = ["SignTestEvaluator"]

import itertools

import numpy as np
import pandas as pd

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer


class SignTestEvaluator(BaseBenchmarkAnalyzer):
    """Sign test for consistent differences between estimator pairs.

    Non-parametric test based on the binomial distribution, counting the number
    of datasets on which one estimator beats another, reproducing the legacy
    ``Evaluator.sign_test``. Computed for all ordered estimator pairs.

    Returns
    -------
    pandas.DataFrame
        Columns ``["estimator_1", "estimator_2", "p_val"]``.
    """

    _tags = {"property:analyzer_type": "pairwise"}

    def _evaluate(self, scores):
        from scipy import stats

        # scipy >= 1.7 uses binomtest; older releases use binom_test
        binom = stats.binomtest if hasattr(stats, "binomtest") else stats.binom_test

        estimator_scores = self._as_estimator_dict(scores)
        rows = []
        for estim_1, estim_2 in itertools.product(estimator_scores.keys(), repeat=2):
            x = np.asarray(estimator_scores[estim_1])
            y = np.asarray(estimator_scores[estim_2])
            n_wins = int(np.sum(x > y))
            p_val = binom(n_wins, len(x)).pvalue
            rows.append(
                {"estimator_1": estim_1, "estimator_2": estim_2, "p_val": p_val}
            )
        return pd.DataFrame(rows, columns=["estimator_1", "estimator_2", "p_val"])
