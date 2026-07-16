"""Average-rank evaluator for benchmark results."""

__all__ = ["RankEvaluator"]

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer


class RankEvaluator(BaseBenchmarkAnalyzer):
    """Average ranks of estimators across datasets.

    Ranks the estimators on every dataset (task) and averages the ranks across
    datasets, reproducing the legacy ``Evaluator.rank``.

    Parameters
    ----------
    metric : str, optional (default=None)
        Metric to analyse; see ``BaseBenchmarkAnalyzer``.
    lower_is_better : bool, optional (default=True)
        If ``True``, the best (lowest) score on each dataset gets rank 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``["model_id", "rank"]``, sorted by ascending average rank.
    """

    _tags = {"property:analyzer_type": "ranking"}

    def _evaluate(self, scores):
        return self._mean_ranks(scores)
