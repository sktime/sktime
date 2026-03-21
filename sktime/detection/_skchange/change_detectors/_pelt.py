"""The pruned exact linear time (PELT) algorithm."""

__author__ = ["Tveten", "johannvk"]
__all__ = ["PELT"]

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..base import BaseIntervalScorer
from ..costs import L2Cost
from ..costs.base import BaseCost
from ..penalties import make_bic_penalty
from ..utils.numba import njit
from ..utils.validation.data import check_data
from ..utils.validation.interval_scorer import check_interval_scorer
from ..utils.validation.parameters import check_larger_than
from ..utils.validation.penalties import check_penalty
from .base import BaseChangeDetector


@dataclass(frozen=True, kw_only=True, eq=False)
class PELTResult:
    """Result of running the PELT algorithm.

    Containing:
    - `optimal_costs`: The optimal penalized segmentation costs for each sample.
    - `previous_change_points`: The optimal previous change point for each sample.
    - `pruning_fraction`: The fraction of starts pruned during the run, as compared
                          to Optimal Partitioning.
    - `changepoints`: The final set of changepoints.
    """

    optimal_costs: np.ndarray
    previous_change_points: np.ndarray
    pruning_fraction: float
    changepoints: np.ndarray

    def __eq__(self, other):
        """Check equality between two PELTResult instances.

        Compares all attributes using numpy's array_equal for array attributes.
        """
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
    def new(
        cls,
        optimal_costs: np.ndarray,
        previous_change_points: np.ndarray,
        pruning_fraction: float,
    ) -> "PELTResult":
        """Create a new PeltResult instance."""
        # Check that the lengths of opt_cost and prev_cpts match:
        if len(optimal_costs) != len(previous_change_points):
            raise ValueError(
                "All input arrays must have the same length. "
                "The lengths of `opt_cost` and `prev_cpts` were "
                f"{len(optimal_costs)} != {len(previous_change_points)}."
            )
        changepoints = get_changepoints(previous_change_points)
        return cls(
            optimal_costs=optimal_costs,
            previous_change_points=previous_change_points,
            pruning_fraction=pruning_fraction,
            changepoints=changepoints,
        )


@njit
def get_changepoints(prev_cpts: np.ndarray) -> np.ndarray:
    changepoints = []
    i = len(prev_cpts) - 1
    while i >= 0:
        cpt_i = prev_cpts[i]
        changepoints.append(cpt_i)
        i = cpt_i - 1
    return np.array(changepoints[-2::-1])  # Remove the artificial changepoint at 0.


def _run_pelt(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm.

    Currently agrees with the 'changepoint::cpt.mean' implementation of PELT in R.
    If the 'min_segment_length' is large enough to span more than a single changepoint,
    the algorithm can return a suboptimal partitioning.
    In that case, resort to the 'optimal_partitioning' algorithm.

    Contract:
    - The `cost` will never be evaluated on intervals shorter than `min_segment_length`.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    min_segment_length : int
        The minimum length of a segment, by default 1.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with a large tolerance.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    cost.check_is_fitted()
    n_samples = cost.n_samples

    if min_segment_length > n_samples:
        raise ValueError(
            "The `min_segment_length` cannot be larger than the number of samples."
        )

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    min_segment_shift = min_segment_length - 1

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1 : min(min_segment_length, n_samples)] = np.inf

    num_pelt_cost_evals = 0
    num_opt_part_cost_evals = 0

    # Compute the optimal cost for indices
    # [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_slice_end = min(2 * min_segment_length, n_samples + 1)
    non_changepoint_ends = np.arange(min_segment_length, non_changepoint_slice_end)
    non_changepoint_starts = np.zeros(len(non_changepoint_ends), dtype=np.int64)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )

    # TODO: Only allow aggregated costs in to "PELT"? User decides aggregation method.
    non_changepoint_costs = np.sum(cost.evaluate(non_changepoint_intervals), axis=1)
    opt_cost[min_segment_length:non_changepoint_slice_end] = non_changepoint_costs

    num_pelt_cost_evals += len(non_changepoint_starts)
    num_opt_part_cost_evals += len(non_changepoint_starts)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    potential_change_point_indices = np.arange(2 * min_segment_length - 1, n_samples)

    # Add a buffer for pruning indices: Start as empty arrays.
    pruning_indices = [np.array([]) for _ in range(min_segment_length)]

    # Triangle number forumla for the unpruned number of cost evaluations:
    num_opt_part_cost_evals += (len(potential_change_point_indices) + 2) * (
        len(potential_change_point_indices) + 1
    ) // 2 - 1

    for current_obs_ind in potential_change_point_indices:
        latest_start = current_obs_ind - min_segment_shift
        opt_cost_obs_ind = current_obs_ind + 1

        if prune:
            starts_to_prune = pruning_indices[current_obs_ind % min_segment_length]
            # Delete the start indices that can be pruned:
            cost_eval_starts = np.delete(
                cost_eval_starts,
                np.where(np.isin(cost_eval_starts, starts_to_prune))[0],
            )

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, np.array([latest_start])))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        interval_costs = np.sum(cost.evaluate(cost_eval_intervals), axis=1)

        num_pelt_cost_evals += len(cost_eval_starts)

        # Add the cost and penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = cost_eval_starts[argmin_candidate_cost]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            # Store indices to prune for the `min_segment_length`'th next observation:
            pruning_indices[current_obs_ind % min_segment_length] = cost_eval_starts[
                candidate_opt_costs > start_inclusion_threshold
            ]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def _run_pelt_min_segment_length_one(
    cost: BaseCost,
    penalty: float,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm, with a minimum segment length of one.

    This is a special case of the PELT algorithm, where the minimum segment length
    is set to one. This means that the algorithm can consider every single point as a
    potential changepoint, and thus it can be used for very fine-grained change point
    detection. We can also simplify the algorithm by not having to keep track of
    deferred pruning information, as all pruning of start points is applicable
    for the next observation.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.
    prune: bool, optional
        If False, drop the pruning step, performing optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    Returns
    -------
    PELTResult
        Summary of the PELT algorithm run, containing:
        - `optimal_cost`: The optimal costs for each segment.
        - `previous_change_points`: The previous changepoints for each segment.
        - `pruning_fraction`: The fraction of pruning applied during the run.
        - `changepoints`: The final set of changepoints.
    """
    cost.check_is_fitted()
    n_samples = cost.n_samples
    if n_samples < 1:
        raise ValueError(
            "The number of samples for the fitted cost must be at least one. "
            f"Got {n_samples} samples."
        )

    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Compute the cost for the first observation directly:
    opt_cost[1] = cost.evaluate(np.array([[0, 1]]))[0, 0]

    # Aggregate number of cost evaluations:
    num_pelt_cost_evals = 1
    num_opt_part_cost_evals = 1

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Evolving set of admissible segment starts.
    eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(1, n_samples)

    num_opt_part_cost_evals += (len(observation_indices) + 2) * (
        len(observation_indices) + 1
    ) // 2 - 1

    for current_obs_ind in observation_indices:
        opt_cost_obs_ind = current_obs_ind + 1

        # Add the next start to the admissible starts set:
        eval_starts = np.concatenate((eval_starts, np.array([current_obs_ind])))
        eval_ends = np.repeat(current_obs_ind + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(eval_intervals), axis=1)

        num_pelt_cost_evals += len(eval_starts)

        # Add the cost and penalty for a new segment:
        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[opt_cost_obs_ind] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = eval_starts[argmin_candidate_cost]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[opt_cost_obs_ind]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            # Apply pruning:
            eval_starts = eval_starts[candidate_opt_costs <= start_inclusion_threshold]

    pruning_fraction = (
        1.0 - num_pelt_cost_evals / num_opt_part_cost_evals
        if num_opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def _run_pelt_with_step_size(
    cost: BaseCost,
    penalty: float,
    step_size: int,
    split_cost: float = 0.0,
    prune: bool = True,
    pruning_margin: float = 0.0,
) -> PELTResult:
    """Run the PELT algorithm.

    Solves the PELT optimization problem where only indices that are multiples of
    `step_size` from the start (index `0`) are considered as potential changepoints.
    This means that the minimum segment length is naturally `step_size`.

    Parameters
    ----------
    cost: BaseCost
        The cost to use.
    penalty : float
        The penalty incurred for adding a changepoint.
    step_size : int
        Only indices that are multiples of `step_size` from the start (index `0`) are
        considered as potential changepoints. This also means that the minimum segment
        length is naturally `step_size`.
    split_cost : float, optional
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the
        above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    Returns
    -------
    PELTResult
        Container for the results of the PELT algorithm run.
    """
    cost.check_is_fitted()
    n_samples = cost.n_samples
    if n_samples < step_size:
        raise ValueError("The `step_size` cannot be larger than the number of samples.")

    # Initialize the optimal costs array:
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    # Initialize to invalid previous changepoints:
    prev_cpts = np.zeros(n_samples, dtype=np.int64)

    # Evolving set of admissible segment starts.
    eval_starts = np.array([], dtype=np.int64)

    observation_interval_starts = np.arange(
        start=0, stop=n_samples - step_size + 1, step=step_size
    )
    observation_interval_ends = np.concatenate(
        (
            np.arange(start=step_size - 1, stop=n_samples - step_size, step=step_size),
            np.array([n_samples - 1]),
        )
    )
    observation_intervals = np.column_stack(
        (observation_interval_starts, observation_interval_ends)
    )

    # Triangle number formula for the unpruned number of cost evaluations.
    opt_part_cost_evals = (
        len(observation_intervals) * (len(observation_intervals) + 1) // 2
    )
    pelt_cost_evals = 0

    for obs_interval_start, obs_interval_end in observation_intervals:
        # Add the next start to the admissible starts set:
        eval_starts = np.concatenate((eval_starts, np.array([obs_interval_start])))
        eval_ends = np.repeat(obs_interval_end + 1, len(eval_starts))
        eval_intervals = np.column_stack((eval_starts, eval_ends))
        interval_costs = np.sum(cost.evaluate(eval_intervals), axis=1)

        pelt_cost_evals += len(eval_starts)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[eval_starts] + interval_costs + penalty

        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        opt_cost[obs_interval_start + 1 : obs_interval_end + 1 + 1] = (
            candidate_opt_costs[argmin_candidate_cost]
        )
        prev_cpts[obs_interval_start : obs_interval_end + 1] = eval_starts[
            argmin_candidate_cost
        ]

        if prune:
            # Trimming the admissible starts set: (reuse the array of optimal costs)
            current_obs_ind_opt_cost = opt_cost[obs_interval_start + 1]

            abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)
            start_inclusion_threshold = (
                current_obs_ind_opt_cost
                # Apply pruning margin to the current optimal cost:
                + abs_current_obs_opt_cost * pruning_margin
                # Moved from 'negative' on left side
                # to 'positive' on right side.
                + penalty
                # Remove from right side of inequality.
                - split_cost
            )

            new_start_inclusion_mask = candidate_opt_costs <= start_inclusion_threshold
            eval_starts = eval_starts[new_start_inclusion_mask]

    pruning_fraction = (
        (1.0 - pelt_cost_evals / opt_part_cost_evals)
        if opt_part_cost_evals > 0
        else np.nan
    )

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


class PELT(BaseChangeDetector):
    """Pruned exact linear time (PELT) changepoint detection.

    Implements the PELT algorithm [1]_ for changepoint detection.
    This method solves the penalized optimal partitioning problem,
    with pruning of the admissible starts set applied to improve performance.

    One can specify a minimum segment length for the partitions considered
    when detecting changepoints through the `min_segment_length` parameter,
    and when the minimum segment length is greater than one we use deferred
    pruning of the admissible starts [2]_ to ensure exact solutions.

    Additionally, one can specify a step size through the `step_size` parameter,
    which coarsens the search space for changepoints, allowing for faster detection
    at the cost of change point location granularity.

    Parameters
    ----------
    cost : BaseIntervalScorer, optional, default=`L2Cost`
        The cost to use for the changepoint detection. Expects a `BaseIntervalScorer`
        instance that implements the `cost` task. If `None`, defaults to `L2Cost`.
    penalty : float, optional
        The penalty to use for the changepoint detection. It must be non-negative. If
        `None`, the penalty is set to
        `make_bic_penalty(n=X.shape[0], n_params=cost.get_model_size(X.shape[1]))`,
        where ``X`` is the input data to `predict` changepoints in.
    min_segment_length : int, optional, default=1
        Minimum length of a segment. The minimum length of a segment to consider
        when detecting changepoints. Must be at least 1. If `step_size` is greater than
        1, this must be less than or equal to `step_size`.
    step_size: bool, optional, default=False
        If True, only indices that are multiples of `step_size` from the
        first data point (index `0`) are considered as potential changepoints.
        Implicitly ensures that `min_segment_length >= step_size`, but it's
        an error to specify `min_segment_length` greater than `step_size`.
    split_cost : float, optional, default=0.0
        The cost of splitting a segment, to ensure that
        cost(X[t:p]) + cost(X[p:(s+1)]) + split_cost <= cost(X[t:(s+1)]),
        for all possible splits, 0 <= t < p < s <= len(X) - 1.
        By default set to 0.0, which is sufficient for
        log likelihood cost functions to satisfy the above inequality.
    prune : bool, optional, default=False
        If True, drop the pruning step. Reverts to optimal partitioning.
        Can be useful for debugging and testing. By default set to False.
    pruning_margin : float, optional, default=0.0
        The pruning margin to use. By default set to zero.
        This is used to reduce pruning of the admissible starts set.
        Can be useful if the cost function is imprecise, i.e.
        based on solving an optimization problem with large tolerance.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American Statistical
    Association, 107(500), 1590-1598.

    .. [2] Bakka, Kristin Benedicte (2018). Changepoint model selection in Gaussian data
    by maximization of approximate Bayes Factors with the Pruned Exact Linear Time
    algorithm. Master's thesis, Norwegian University of Science and Technology (NTNU).
    URL: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2558597.

    Examples
    --------
    >>> from sktime.detection._skchange.change_detectors import PELT
    >>> from sktime.detection._skchange.datasets import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=2, mean=10, segment_length=100, p=5)
    >>> detector = PELT()
    >>> detector.fit_predict(df)
       ilocs
    0    100
    """

    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": ["Tveten", "johannvk"],
        "fit_is_empty": True,
    }

    def __init__(
        self,
        cost: BaseIntervalScorer | None = None,
        penalty: float | None = None,
        min_segment_length: int = 1,
        step_size: int = 1,
        split_cost: float = 0.0,
        prune: bool = True,
        pruning_margin: float = 0.0,
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
        self._pelt_result: PELTResult | None = None
        self.fitted_cost: BaseCost | None = None
        self.fitted_penalty: float | None = None

        check_penalty(
            penalty,
            "penalty",
            "PELT",
            require_constant_penalty=True,
            allow_none=True,
        )
        check_larger_than(1, min_segment_length, "min_segment_length")

        self.clone_tags(self._cost, ["distribution_type"])

    def _fit_cost_and_penalty(
        self,
        X: pd.DataFrame | pd.Series,
    ):
        X = check_data(
            X,
            min_length=2 * self.min_segment_length,
            min_length_name="2*min_segment_length",
        )

        self.fitted_cost = self._cost.clone()
        self.fitted_cost.fit(X)

        if self.penalty is None:
            self.fitted_penalty = make_bic_penalty(
                n=X.shape[0],
                n_params=self.fitted_cost.get_model_size(X.shape[1]),
            )
        else:
            self.fitted_penalty = self.penalty

    def _predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Detect events in test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect change points in.

        Returns
        -------
        y_sparse : pd.DataFrame
            A `pd.DataFrame` with a range index and one column:
            * ``"ilocs"`` - integer locations of the changepoints.

        Attributes
        ----------
        fitted_cost : BaseIntervalScorer
            The fitted cost function.
        fitted_penalty : float
            The fitted penalty value. Either the user-specified value or the fitted BIC
            penalty.
        """
        self._fit_cost_and_penalty(X)

        if self.step_size > 1:
            # If step_size > 1, use the JumpPELT algorithm:
            pelt_result = _run_pelt_with_step_size(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                step_size=self.step_size,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        elif self.min_segment_length == 1:
            # Special case for min_segment_length=1, with less overhead:
            pelt_result = _run_pelt_min_segment_length_one(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )
        else:
            pelt_result = _run_pelt(
                cost=self.fitted_cost,
                penalty=self.fitted_penalty,
                min_segment_length=self.min_segment_length,
                split_cost=self.split_cost,
                prune=self.prune,
                pruning_margin=self.pruning_margin,
            )

        # Store the scores for introspection without recomputing using transform_scores
        self._pelt_result = pelt_result
        self.scores = pd.Series(pelt_result.optimal_costs, index=X.index, name="score")
        return self._format_sparse_output(pelt_result.changepoints)

    def _transform_scores(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to score (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence `X`.

        Notes
        -----
        The PELT scores are the cumulative optimal costs, so the scores are increasing
        and are not per observation scores.
        """
        self.predict(X)
        return self.scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.detection._skchange.costs import L2Cost

        params = [
            {"cost": L2Cost(), "min_segment_length": 5},
            {"cost": L2Cost(), "penalty": 0.0, "min_segment_length": 4, "step_size": 4},
        ]
        return params
