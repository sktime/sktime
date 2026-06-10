# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Change score adapter that wraps any cost into a change score."""

__author__ = ["Tveten"]


from sktime.detection._utils import check_interval_scorer
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


class ChangeScore(BaseIntervalScorer):
    """Change score constructed from a cost function.

    Evaluates the reduction in cost obtained by allowing a split within
    an interval, i.e. ``cost(total) - (cost(left) + cost(right))``.

    Parameters
    ----------
    cost : BaseCost
        An interval scorer with ``task="cost"``.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "change_score",
    }

    def __init__(self, cost):
        self.cost = cost
        super().__init__()
        check_interval_scorer(self.cost, "cost", "ChangeScore", required_tasks=["cost"])
        if self.cost.get_tag("is_aggregated"):
            self.set_tags(**{"is_aggregated": True})

    @property
    def min_size(self):
        return self.cost.min_size

    def get_model_size(self, p):
        """Get the number of parameters per segment."""
        return self.cost.get_model_size(p)

    def _evaluate(self, X, cuts):
        """Evaluate the cost-based change score.

        Parameters
        ----------
        X : np.ndarray
            2D data array.
        cuts : np.ndarray
            3-column array ``[start, split, end]``.

        Returns
        -------
        scores : np.ndarray
            2D array of change scores.
        """
        total = cuts[:, [0, 2]]
        left = cuts[:, [0, 1]]
        right = cuts[:, [1, 2]]
        return self.cost.evaluate(X, total) - (
            self.cost.evaluate(X, left) + self.cost.evaluate(X, right)
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._l2_cost import L2Cost

        return [{"cost": L2Cost()}]


def to_change_score(scorer):
    """Convert a compatible scorer to a change score.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        A cost-task scorer or an existing change score.

    Returns
    -------
    BaseIntervalScorer
        A ``ChangeScore`` wrapping the cost, or the scorer itself if
        already a change score.
    """
    from sktime.detection.base._base_interval_scorer import BaseIntervalScorer

    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(f"scorer must be a BaseIntervalScorer. Got {type(scorer)}.")
    task = scorer.get_tag("task")
    if task == "cost":
        return ChangeScore(cost=scorer)
    elif task == "change_score":
        return scorer
    raise ValueError(
        f"scorer must have tag 'task' equal to 'cost' or 'change_score'. "
        f"Got scorer.get_tag('task') = {task}."
    )
