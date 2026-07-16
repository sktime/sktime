"""Average-rank evaluator for benchmark results."""

__all__ = ["RankEvaluator"]

from sktime.benchmarking.post_hoc._base import BasePostHocEvaluator


class RankEvaluator(BasePostHocEvaluator):
    """Average ranks of estimators across datasets.

    Ranks the estimators on every dataset (task) and averages the ranks across
    datasets, reproducing the legacy ``Evaluator.rank``.

    Parameters
    ----------
    metric : str, optional (default=None)
        Metric to analyse; see ``BasePostHocEvaluator``.
    lower_is_better : bool, optional (default=True)
        If ``True``, the best (lowest) score on each dataset gets rank 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``["model_id", "rank"]``, sorted by ascending average rank.
    """

    _tags = {"property:evaluator_type": "ranking"}

    def _evaluate(self, scores):
        return self._mean_ranks(scores)
