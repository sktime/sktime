"""Cost-based change scores."""

import numpy as np

from ..base import BaseIntervalScorer
from ..costs.base import BaseCost


def to_change_score(scorer: BaseIntervalScorer) -> BaseIntervalScorer:
    """Convert compatible scorers to a change score.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The scorer to convert to a change score. If a change score is provided, it is
        returned as is.

    Returns
    -------
    BaseIntervalScorer
        The change score.
    """
    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(f"scorer must be a BaseIntervalScorer. Got {type(scorer)}.")
    task = scorer.get_tag("task")
    if task == "cost":
        change_score = ChangeScore(scorer)
    elif task == "change_score":
        change_score = scorer
    else:
        raise ValueError(
            f"scorer must have tag 'task' set to 'cost' or 'change_score'."
            f" Got scorer.get_tag('task') = {task}."
        )
    return change_score


class ChangeScore(BaseIntervalScorer):
    """Change score based on a cost class.

    The change score is calculated as cost difference under a no-change hypothesis
    versus a single change hypothesis.

    Parameters
    ----------
    cost : BaseCost
        The cost function.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "change_score",
    }

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

        self.clone_tags(
            cost,
            ["distribution_type", "is_conditional", "is_aggregated", "is_penalised"],
        )

    @property
    def min_size(self) -> int | None:
        """Minimum valid size of an interval to evaluate."""
        if self.is_fitted:
            return self.cost_.min_size
        else:
            return self.cost.min_size

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters to estimate over each interval.

        The primary use of this method is to determine an appropriate default penalty
        value in detectors.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        if self.is_fitted:
            return self.cost_.get_model_size(p)
        else:
            return self.cost.get_model_size(p)

    def _fit(self, X: np.ndarray, y=None):
        """Fit the change score.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self.cost_: BaseCost = self.cost.clone()
        self.cost_.fit(X)
        return self

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the change score for a split within an interval.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate.
            The difference between subsets ``X[start:split]`` and ``X[split:end]`` is
            evaluated for each row in `cuts`.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        left_intervals = cuts[:, [0, 1]]
        right_intervals = cuts[:, [1, 2]]
        full_intervals = cuts[:, [0, 2]]
        left_costs = self.cost_.evaluate(left_intervals)
        right_costs = self.cost_.evaluate(right_intervals)
        no_change_costs = self.cost_.evaluate(full_intervals)

        change_scores = no_change_costs - (left_costs + right_costs)

        # Ensure that slightly negative scores are set to 0:
        change_scores[(change_scores < 0) & (change_scores > -1e-8)] = 0.0

        return change_scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.detection._skchange.costs import GaussianCost, L2Cost

        params = [
            {"cost": L2Cost()},
            {"cost": GaussianCost()},
        ]
        return params
