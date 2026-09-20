"""Friedman omnibus test evaluator for benchmark results."""

__all__ = ["FriedmanTest"]

import pandas as pd

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer


class FriedmanTest(BaseBenchmarkAnalyzer):
    """Friedman test for differences between estimators across datasets.

    Non-parametric omnibus test of the null hypothesis that all estimators
    perform equally across datasets, reproducing the legacy
    ``Evaluator.friedman_test`` (``scipy.stats.friedmanchisquare``). A
    significant result is the usual precondition for running a post-hoc test
    such as ``NemenyiTest``.

    Returns
    -------
    pandas.DataFrame
        Single row with columns ``["statistic", "p_value"]``.
    """

    _tags = {"property:analyzer_type": "omnibus"}

    def _evaluate(self, scores):
        from scipy.stats import friedmanchisquare

        estimator_scores = self._as_estimator_dict(scores)
        statistic, p_value = friedmanchisquare(*estimator_scores.values())
        return pd.DataFrame([[statistic, p_value]], columns=["statistic", "p_value"])
