# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Saving-type anomaly scores from cost functions."""

__author__ = ["Tveten"]

import numpy as np

from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def to_saving(scorer):
    """Convert a compatible scorer to a saving.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        A cost-task scorer (with fixed baseline parameter) or an existing saving.

    Returns
    -------
    BaseIntervalScorer
        A ``Saving`` wrapping the cost, or the scorer itself if already a saving.
    """
    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(f"scorer must be a BaseIntervalScorer. Got {type(scorer)}.")
    task = scorer.get_tag("task")
    if task == "cost":
        return Saving(scorer)
    elif task == "saving":
        return scorer
    raise ValueError(
        f"scorer must have tag 'task' equal to 'cost' or 'saving'. "
        f"Got scorer.get_tag('task') = {task}."
    )


class Saving(BaseIntervalScorer):
    """Saving based on a cost class.

    The saving is the difference between the cost of a fixed baseline parameter
    and the optimised (MLE) cost over a given interval. The baseline parameter
    should be robustly estimated across the entire dataset. Each saving
    indicates the potential cost reduction if the parameter were optimised
    for that interval.

    Parameters
    ----------
    baseline_cost : BaseCost
        A cost with a fixed ``param`` set. The optimised cost is derived by
        cloning with ``param=None``.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "saving",
    }

    def __init__(self, baseline_cost):
        self.baseline_cost = baseline_cost
        super().__init__()

        if not baseline_cost.get_tag("supports_fixed_param"):
            raise ValueError(
                "The baseline cost must support fixed parameter(s). "
                "Indicated by the tag 'supports_fixed_param'."
            )
        if baseline_cost.param is None:
            raise ValueError(
                "The baseline cost must have a fixed parameter set "
                "(``param`` must not be ``None``)."
            )
        self._optimised_cost = baseline_cost.clone().set_params(param=None)

    @property
    def min_size(self):
        return self._optimised_cost.min_size

    def get_model_size(self, p):
        return self._optimised_cost.get_model_size(p)

    def _evaluate(self, X, cuts):
        baseline_costs = self.baseline_cost.evaluate(X, cuts)
        optimised_costs = self._optimised_cost.evaluate(X, cuts)
        return baseline_costs - optimised_costs

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {"baseline_cost": L2Cost(param=0.0)},
            {"baseline_cost": L2Cost(param=np.array([1.0]))},
        ]


def to_local_anomaly_score(scorer):
    """Convert a compatible scorer to a local anomaly score.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        A cost-task scorer or an existing local anomaly score.

    Returns
    -------
    BaseIntervalScorer
        A ``LocalAnomalyScore`` wrapping the cost, or the scorer itself.
    """
    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(f"scorer must be a BaseIntervalScorer. Got {type(scorer)}.")
    task = scorer.get_tag("task")
    if task == "cost":
        return LocalAnomalyScore(scorer)
    elif task == "local_anomaly_score":
        return scorer
    raise ValueError(
        f"scorer must have tag 'task' equal to 'cost' or 'local_anomaly_score'. "
        f"Got scorer.get_tag('task') = {task}."
    )


class LocalAnomalyScore(BaseIntervalScorer):
    """Local anomaly scores based on a cost function.

    Compares the cost behaviour of an inner interval with the surrounding data
    in an outer interval. The null hypothesis within each outer interval is
    stationarity; the alternative is a segment anomaly.

    Parameters
    ----------
    cost : BaseCost
        The cost function to evaluate data subsets.

    Notes
    -----
    Using costs to generate local anomaly scores is slower than direct anomaly
    scores because evaluating the cost on disjoint surrounding segments
    requires concatenation and re-evaluation per cut.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "local_anomaly_score",
    }

    def __init__(self, cost):
        self.cost = cost
        super().__init__()

    @property
    def min_size(self):
        return self.cost.min_size

    def get_model_size(self, p):
        return self.cost.get_model_size(p)

    def _evaluate(self, X, cuts):
        inner_intervals = cuts[:, 1:3]
        outer_intervals = cuts[:, [0, 3]]
        inner_costs = self.cost.evaluate(X, inner_intervals)
        outer_costs = self.cost.evaluate(X, outer_intervals)

        surrounding_costs = np.zeros_like(outer_costs)
        for i in range(cuts.shape[0]):
            before_data = X[cuts[i, 0] : cuts[i, 1]]
            after_data = X[cuts[i, 2] : cuts[i, 3]]
            surrounding_data = np.concatenate((before_data, after_data))
            n = surrounding_data.shape[0]
            surrounding_costs[i] = self.cost.evaluate(
                surrounding_data, np.array([[0, n]])
            )

        return outer_costs - (inner_costs + surrounding_costs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._gaussian_cost import GaussianCost
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {"cost": L2Cost()},
            {"cost": GaussianCost()},
        ]
