"""Nemenyi post-hoc test evaluator for benchmark results."""

__all__ = ["NemenyiEvaluator"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.benchmarking.evaluators._base import BasePostHocEvaluator


class NemenyiEvaluator(BasePostHocEvaluator):
    """Nemenyi post-hoc test for pairwise differences between estimators.

    Pairwise post-hoc test, typically run after a significant
    ``FriedmanEvaluator`` result, reproducing the legacy ``Evaluator.nemenyi``
    (``scikit_posthocs.posthoc_nemenyi``).

    Returns
    -------
    pandas.DataFrame
        Square matrix of pairwise p-values, indexed and columned by
        ``model_id``.
    """

    _tags = {
        "python_dependencies": "scikit_posthocs",
        "capability:pairwise": True,
    }

    def _evaluate(self, scores):
        _check_soft_dependencies("scikit_posthocs")

        from scikit_posthocs import posthoc_nemenyi

        estimator_scores = self._as_estimator_dict(scores)
        melted = pd.DataFrame(estimator_scores).melt(
            var_name="groups", value_name="values"
        )
        return posthoc_nemenyi(melted, val_col="values", group_col="groups")
