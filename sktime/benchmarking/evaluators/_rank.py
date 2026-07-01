"""Average-rank evaluator for benchmark results."""

__all__ = ["RankEvaluator"]

from sktime.benchmarking.evaluators._base import BasePostHocEvaluator


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

    def _evaluate(self, scores):
        # rank estimators within each dataset (row), then average over datasets
        ranks = scores.rank(axis=1, ascending=self.lower_is_better)
        mean_ranks = ranks.mean(axis=0).reset_index()
        mean_ranks.columns = ["model_id", "rank"]
        return mean_ranks.sort_values("rank").reset_index(drop=True)
