# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten, johannvk
"""Pruned Exact Linear Time (PELT) changepoint detection algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

from dataclasses import dataclass

import numpy as np

from sktime.detection._costs._l2_cost import L2Cost
from sktime.detection._formatters import format_changepoints
from sktime.detection._penalties import make_bic_penalty
from sktime.detection._utils import (
    as_2d_array,
    check_data,
    check_interval_scorer,
    check_larger_than,
    check_penalty,
)
from sktime.detection.base import BaseDetector

# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PELTResult:
    """Result of running the PELT algorithm."""

    optimal_costs: np.ndarray
    previous_change_points: np.ndarray
    pruning_fraction: float
    changepoints: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, PELTResult):
            return False
        return (
            np.array_equal(self.optimal_costs, other.optimal_costs)
            and np.array_equal(
                self.previous_change_points, other.previous_change_points
            )
            and self.pruning_fraction == other.pruning_fraction
            and np.array_equal(self.changepoints, other.changepoints)
        )

    @classmethod
    def new(cls, optimal_costs, previous_change_points, pruning_fraction):
        """Create a new PELTResult with changepoints back-tracked."""
        if len(optimal_costs) != len(previous_change_points):
            raise ValueError(
                "All input arrays must have the same length. "
                f"Got {len(optimal_costs)} != {len(previous_change_points)}."
            )
        changepoints = _get_changepoints(previous_change_points)
        return cls(
            optimal_costs=optimal_costs,
            previous_change_points=previous_change_points,
            pruning_fraction=pruning_fraction,
            changepoints=changepoints,
        )


def _get_changepoints(prev_cpts):
    """Back-track changepoints from the previous-changepoint array."""
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(cpt_i)
        i = int(cpt_i) - 1
    return np.array(changepoints[-2::-1])


def _run_pelt(
    cost,
    X,
    penalty,
    min_segment_length,
    split_cost=0.0,
    prune=True,
    pruning_margin=0.0,
):
    """Run the PELT algorithm (stateless).

    Parameters
    ----------
    cost : BaseCost
        Cost scorer.
    X : np.ndarray
        2D data array.
    penalty : float
        Penalty for adding a changepoint.
    min_segment_length : int
        Minimum segment length.
    split_cost : float
        Cost of splitting.
    prune : bool
        Whether to prune admissible starts.
    pruning_margin : float
        Pruning margin.

    Returns
    -------
    PELTResult
    """
    n_samples = X.shape[0]

    if min_segment_length > n_samples:
        raise ValueError(
            "The `min_segment_length` cannot be larger than the number of samples."
        )

    prev_cpts = np.repeat(0, n_samples)
    min_segment_shift = min_segment_length - 1

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))
    opt_cost[1 : min(min_segment_length, n_samples)] = np.inf

    num_pelt_cost_evals = 0
    num_opt_part_cost_evals = 0

    non_changepoint_slice_end = min(2 * min_segment_length, n_samples + 1)
    non_changepoint_ends = np.arange(min_segment_length, non_changepoint_slice_end)
    non_changepoint_starts = np.zeros(len(non_changepoint_ends), dtype=np.int64)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )

    non_changepoint_costs = np.sum(cost.evaluate(X, non_changepoint_intervals), axis=1)
    opt_cost[min_segment_length:non_changepoint_slice_end] = non_changepoint_costs

    num_pelt_cost_evals += len(non_changepoint_starts)
    num_opt_part_cost_evals += len(non_changepoint_starts)

    cost_eval_starts = np.array([0], dtype=np.int64)

    potential_change_point_indices = np.arange(2 * min_segment_length - 1, n_samples)

    pruning_indices = [np.array([]) for _ in range(min_segment_length)]

    num_opt_part_cost_evals += (len(potential_change_point_indices) + 2) * (
        len(potential_change_point_indices) + 1
    ) // 2 - 1

    for current_obs_ind in potential_change_point_indices:
        latest_start = current_obs_ind - min_segment_shift
        opt_cost_obs_ind = current_obs_ind + 1

        if prune:
            starts_to_prune = pruning_indices[current_obs_ind % min_segment_length]
            cost_eval_starts = np.delete(
                cost_eval_starts,
                np.where(np.isin(cost_eval_starts, starts_to_prune))[0],
            )

        cost_eval_starts = np.concatenate((cost_eval_starts, np.array([latest_start])))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        interval_costs = np.sum(cost.evaluate(X, cost_eval_intervals), axis=1)

        num_pelt_cost_evals += len(cost_eval_starts)

        candidate_opt_costs = opt_cost[cost_eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        if prune:
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]
            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * pruning_margin
                + penalty
                - split_cost
            )
            pruning_indices[current_obs_ind % min_segment_length] = cost_eval_starts[
                candidate_opt_costs > start_inclusion_threshold
            ]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    return PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )


def _run_pelt_min_segment_length_one(
    cost,
    X,
    penalty,
    split_cost=0.0,
    prune=True,
    pruning_margin=0.0,
):
    """Run PELT with min_segment_length=1 (simplified, less overhead)."""
    n_samples = X.shape[0]
    if n_samples < 1:
        raise ValueError(f"Need at least one sample, got {n_samples}.")

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))
    opt_cost[1] = cost.evaluate(X, np.array([[0, 1]]))[0, 0]

    num_pelt_cost_evals = 1
    num_opt_part_cost_evals = 1

    prev_cpts = np.repeat(0, n_samples)
    eval_starts = np.array([0], dtype=np.int64)

    observation_indices = np.arange(1, n_samples)

    num_opt_part_cost_evals += (len(observation_indices) + 2) * (
        len(observation_indices) + 1
    ) // 2 - 1

    for current_obs_ind in observation_indices:
        opt_cost_obs_ind = current_obs_ind + 1

        eval_starts = np.concatenate((eval_starts, np.array([current_obs_ind])))
        eval_ends = np.repeat(current_obs_ind + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(X, eval_intervals), axis=1)

        num_pelt_cost_evals += len(eval_starts)

        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = eval_starts[argmin_candidate_cost]

        if prune:
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]
            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * pruning_margin
                + penalty
                - split_cost
            )
            eval_starts = eval_starts[candidate_opt_costs <= start_inclusion_threshold]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    return PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )


def _run_pelt_with_step_size(
    cost,
    X,
    penalty,
    step_size,
    split_cost=0.0,
    prune=True,
    pruning_margin=0.0,
):
    """Run JumpPELT: PELT with step-size > 1."""
    n_samples = X.shape[0]
    if n_samples < step_size:
        raise ValueError("The `step_size` cannot be larger than the number of samples.")

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))
    prev_cpts = np.zeros(n_samples, dtype=np.int64)
    eval_starts = np.array([], dtype=np.int64)

    observation_interval_starts = np.arange(0, n_samples - step_size + 1, step_size)
    observation_interval_ends = np.concatenate(
        (
            np.arange(step_size - 1, n_samples - step_size, step_size),
            np.array([n_samples - 1]),
        )
    )
    observation_intervals = np.column_stack(
        (observation_interval_starts, observation_interval_ends)
    )

    opt_part_cost_evals = (
        len(observation_intervals) * (len(observation_intervals) + 1) // 2
    )
    pelt_cost_evals = 0

    for obs_interval_start, obs_interval_end in observation_intervals:
        eval_starts = np.concatenate((eval_starts, np.array([obs_interval_start])))
        eval_ends = np.repeat(obs_interval_end + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(X, eval_intervals), axis=1)

        pelt_cost_evals += len(eval_starts)

        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[obs_interval_start + 1 : obs_interval_end + 2] = candidate_opt_costs[
            argmin_candidate_cost
        ]
        prev_cpts[obs_interval_start : obs_interval_end + 1] = eval_starts[
            argmin_candidate_cost
        ]

        if prune:
            current_obs_ind_opt_cost = opt_cost[obs_interval_start + 1]
            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * pruning_margin
                + penalty
                - split_cost
            )
            eval_starts = eval_starts[candidate_opt_costs <= start_inclusion_threshold]

    pruning_fraction = (
        (1.0 - pelt_cost_evals / opt_part_cost_evals)
        if opt_part_cost_evals > 0
        else np.nan
    )

    return PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )


# ---------------------------------------------------------------------------
# PELT Detector
# ---------------------------------------------------------------------------


class PELT(BaseDetector):
    """Pruned Exact Linear Time (PELT) changepoint detection.

    PELT [1]_ solves the penalised optimal-partitioning problem with
    pruning of admissible start points, giving exact solutions in expected
    linear time for well-behaved cost functions.

    Parameters
    ----------
    cost : BaseIntervalScorer, optional, default=L2Cost()
        Cost scorer used for changepoint detection.
    penalty : float or None, default=None
        Penalty for adding a changepoint. ``None`` uses a BIC penalty
        computed from the data at prediction time.
    min_segment_length : int, default=1
        Minimum number of observations in a segment.
    step_size : int, default=1
        Only multiples of ``step_size`` are considered as changepoints
        (JumpPELT).
    split_cost : float, default=0.0
        Additive cost of splitting a segment.
    prune : bool, default=True
        If ``False``, disables pruning (optimal partitioning).
    pruning_margin : float, default=0.0
        Margin added to the pruning criterion.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal
       detection of changepoints with a linear computational cost.
       Journal of the American Statistical Association, 107(500), 1590-1598.

    .. [2] Bakka, K. B. (2018). Changepoint model selection in Gaussian data.
       Master's thesis, NTNU.

    Examples
    --------
    >>> from sktime.detection._pelt import PELT
    >>> from sktime.detection._costs._l2_cost import L2Cost
    >>> import numpy as np
    >>> X = np.concatenate([np.zeros(50), 10 * np.ones(50)])
    >>> detector = PELT(cost=L2Cost(), penalty=15)
    >>> detector.fit(X).predict(X)  # doctest: +SKIP
    """

    _tags = {
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": True,
        "authors": ["Tveten", "johannvk"],
        "maintainers": ["Tveten", "johannvk"],
        "python_dependencies": "numba",
    }

    def __init__(
        self,
        cost=None,
        penalty=None,
        min_segment_length=1,
        step_size=1,
        split_cost=0.0,
        prune=True,
        pruning_margin=0.0,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_segment_length = min_segment_length
        self.step_size = step_size
        self.split_cost = split_cost
        self.prune = prune
        self.pruning_margin = pruning_margin
        super().__init__()

        if self.step_size > 1 and self.min_segment_length > self.step_size:
            raise ValueError(
                f"PELT `min_segment_length`(={self.min_segment_length}) cannot be "
                f"greater than the `step_size`(={self.step_size}) > 1."
            )

        _cost = L2Cost() if cost is None else cost
        check_interval_scorer(
            _cost,
            arg_name="cost",
            caller_name="PELT",
            required_tasks=["cost"],
            allow_penalised=False,
        )
        self._cost = _cost.clone()

        check_penalty(
            penalty,
            "penalty",
            "PELT",
            allow_none=True,
        )
        check_larger_than(1, min_segment_length, "min_segment_length")

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        """Detect changepoints in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Time series (already converted by ``BaseDetector.predict``).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``"ilocs"`` column of integer changepoint positions.
        """
        X_df = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )
        X_arr = as_2d_array(X_df)
        n, p = X_arr.shape

        if self.penalty is None:
            fitted_penalty = make_bic_penalty(
                n_params=self._cost.get_model_size(p), n=n
            )
        else:
            fitted_penalty = self.penalty

        if self.step_size > 1:
            pelt_result = _run_pelt_with_step_size(
                cost=self._cost,
                X=X_arr,
                penalty=fitted_penalty,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        elif self.min_segment_length == 1:
            pelt_result = _run_pelt_min_segment_length_one(
                cost=self._cost,
                X=X_arr,
                penalty=fitted_penalty,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        else:
            pelt_result = _run_pelt(
                cost=self._cost,
                X=X_arr,
                penalty=fitted_penalty,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )

        return format_changepoints(pelt_result.changepoints)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection._costs._l2_cost import L2Cost

        return [
            {"cost": L2Cost(), "min_segment_length": 5},
            {
                "cost": L2Cost(),
                "penalty": 0.0,
                "min_segment_length": 4,
                "step_size": 4,
            },
        ]
