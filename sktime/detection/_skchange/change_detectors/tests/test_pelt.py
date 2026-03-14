"""Tests for the PELT implementation."""

import re
import time

import numpy as np
import pytest

rpt = pytest.importorskip("ruptures")
rpt_BaseCost = rpt.base.BaseCost

from sktime.detection._skchange.change_detectors._crops import evaluate_segmentation
from sktime.detection._skchange.change_detectors._pelt import (
    PELT,
    PELTResult,
    _run_pelt,
    _run_pelt_min_segment_length_one,
    _run_pelt_with_step_size,
    get_changepoints,
)
from sktime.detection._skchange.change_scores import CUSUM
from sktime.detection._skchange.costs import GaussianCost, L2Cost
from sktime.detection._skchange.costs.base import BaseCost
from sktime.detection._skchange.datasets import generate_alternating_data

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=20, segment_length=seg_len, p=1, random_state=2
).values.reshape(-1, 1)

alternating_sequence = generate_alternating_data(
    n_segments=5,
    mean=10.5,
    variance=0.5,
    segment_length=21,
    p=1,
    random_state=5,
).values.reshape(-1, 1)

long_alternating_sequence = generate_alternating_data(
    n_segments=30,
    mean=10.5,
    variance=0.5,
    segment_length=21,
    p=1,
    random_state=5,
).values.reshape(-1, 1)


@pytest.fixture
def cost():
    """Generate a new cost object for each test."""
    cost = L2Cost()
    return cost


@pytest.fixture
def penalty() -> float:
    """Penalty for the PELT algorithm."""
    penalty = 2 * np.log(len(changepoint_data))
    return penalty


class RupturesGaussianCost(rpt_BaseCost):
    """Custom cost for Gaussian (mean-var) cost."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal) -> "RupturesGaussianCost":
        self.signal = signal
        self.cost = GaussianCost().fit(signal)
        return self

    def error(self, start: int, end: int) -> np.ndarray:
        """Compute the cost of a segment."""
        cuts = np.array([start, end]).reshape(-1, 2)
        return self.cost.evaluate(cuts)


def pelt_partition_cost(
    X: np.ndarray,
    changepoints: np.ndarray,
    cost: BaseCost,
    penalty: float,
):
    cost.fit(X)
    num_samples = len(X)

    # Add number of 'segments' * penalty to the cost.
    # Instead of number of 'changepoints' * penalty.
    # total_cost = penalty * (len(changepoints) + 1)
    total_cost = penalty * len(changepoints)
    np_changepoints = np.asarray(changepoints, dtype=np.int64)

    interval_starts = np.concatenate((np.array([0]), np_changepoints), axis=0)
    interval_ends = np.concatenate((np_changepoints, np.array([num_samples])), axis=0)

    interval_costs = np.sum(
        cost.evaluate(np.column_stack((interval_starts, interval_ends))), axis=1
    )
    total_cost += np.sum(interval_costs)

    return total_cost


def old_run_pelt(
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
        log likelihood cost functions to satisfy the
        above inequality.
    prune: bool, optional
        If False, drop the pruning step, reverting to optimal partitioning.
        Can be useful for debugging and testing. By default set to True.
    pruning_margin : float, optional
        The percentage of pruning margin to use. By default set to zero.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.

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
    min_segment_shift = min_segment_length - 1

    # Redefine Opt_cost[0] to start at 0.0, as done in 2014 PELT.
    # opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))
    opt_cost = np.concatenate((np.array([-penalty]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = np.inf

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs

    # Aggregate number of cost evaluations:
    num_pelt_cost_evals = len(non_changepoint_starts)
    num_opt_part_cost_evals = len(non_changepoint_starts)

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Evolving set of admissible segment starts.
    cost_eval_starts = np.array(([0]), dtype=np.int64)

    observation_indices = np.arange(2 * min_segment_length - 1, n_samples).reshape(
        -1, 1
    )

    num_opt_part_cost_evals += int(
        (len(observation_indices) + 2) * (len(observation_indices) + 1) // 2 - 1
    )

    for current_obs_ind in observation_indices:
        latest_start = current_obs_ind - min_segment_shift
        opt_cost_obs_ind = current_obs_ind[0] + 1

        # Add the next start to the admissible starts set:
        cost_eval_starts = np.concatenate((cost_eval_starts, latest_start))
        cost_eval_ends = np.repeat(current_obs_ind + 1, len(cost_eval_starts))
        cost_eval_intervals = np.column_stack((cost_eval_starts, cost_eval_ends))
        costs = cost.evaluate(cost_eval_intervals)
        agg_costs = np.sum(costs, axis=1)

        num_pelt_cost_evals += len(cost_eval_starts)

        # Add the penalty for a new segment:
        candidate_opt_costs = opt_cost[cost_eval_starts] + agg_costs + penalty

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

            # Apply pruning:
            cost_eval_starts = cost_eval_starts[
                candidate_opt_costs <= start_inclusion_threshold
            ]

    pruning_fraction = 1.0 - num_pelt_cost_evals / num_opt_part_cost_evals

    pelt_result = PELTResult.new(
        optimal_costs=opt_cost[1:],
        previous_change_points=prev_cpts,
        pruning_fraction=pruning_fraction,
    )

    return pelt_result


def run_pelt_masked(
    cost: BaseCost,
    penalty: float,
    min_segment_length: int,
    split_cost: float = 0.0,
    pruning_margin: float = 0.0,
    allocation_multiplier: float = 5.0,  # Initial multiple of log(n_samples)
    growth_factor: float = 2.0,  # Geometric growth factor
) -> tuple[np.ndarray, list]:
    """Run the PELT algorithm.

    Currently agrees with the 'changepoint::cpt.mean' implementation of PELT in R.
    If the 'min_segment_length' is large enough to span more than a single changepoint,
    the algorithm can return a suboptimal partitioning.
    In that case, resort to the 'optimal_partitioning' algorithm.

    # Noteworthy: Uses pre-allocated arrays as much as possible,
    # to avoid excessive memory allocation and copying.

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
        log likelihood cost functions to satisfy the
        above inequality.
    pruning_margin : float, optional
        The pruning margin to use. By default set to zero.
        This is used to prune the admissible starts set.
        The pruning margin is used to avoid numerical issues when comparing
        the candidate optimal costs with the current optimal cost.
    initial_capacity : float, optional
        The initial capacity of the pre-allocated arrays.
        This is a multiple of log(n_samples). Default is 5.0.
    growth_factor : float, optional
        The factor by which to grow the arrays when they need to be resized.
        Default is 2.0.

    Returns
    -------
    tuple[np.ndarray, list, float]
        The optimal costs, the changepoints, and cost evaluation time.
    """
    cost.check_is_fitted()
    n_samples = cost.n_samples
    min_segment_shift = min_segment_length - 1

    # Explicitly set the first element to 0.
    # Define "opt_cost[0]"" to start at 0.0, as done in 2014 PELT.
    opt_cost = np.concatenate((np.array([0.0]), np.zeros(n_samples)))

    # Cannot compute the cost for the first 'min_segment_shift' elements:
    opt_cost[1:min_segment_length] = 0.0

    # Compute the cost in [min_segment_length, 2*min_segment_length - 1] directly:
    non_changepoint_starts = np.zeros(min_segment_length, dtype=np.int64)
    non_changepoint_ends = np.arange(min_segment_length, 2 * min_segment_length)
    non_changepoint_intervals = np.column_stack(
        (non_changepoint_starts, non_changepoint_ends)
    )
    costs = cost.evaluate(non_changepoint_intervals)
    agg_costs = np.sum(costs, axis=1)
    opt_cost[min_segment_length : 2 * min_segment_length] = agg_costs + penalty

    # Store the previous changepoint for each latest start added.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = np.repeat(0, n_samples)

    # Initialize smaller arrays with a fraction of n_samples capacity
    initial_size = max(2, int(np.log(n_samples) * allocation_multiplier))

    # Pre-allocate arrays with initial capacity
    start_capacity = initial_size
    starts_buffer = np.zeros(start_capacity, dtype=np.int64)
    interval_capacity = initial_size
    interval_buffer = np.zeros((interval_capacity, 2), dtype=np.int64)

    # Initialize with the first valid start position (position 0)
    n_valid_starts = 1
    starts_buffer[0] = 0  # First valid start is at position 0
    cost_eval_time = 0.0

    for current_obs_ind in range(2 * min_segment_length - 1, n_samples):
        latest_start = current_obs_ind - min_segment_shift

        # Add the next start position to the admissible set:
        # First check if we need to grow the arrays
        if n_valid_starts + 1 > start_capacity:
            # Grow arrays geometrically
            new_capacity = int(start_capacity * growth_factor)
            new_starts_buffer = np.zeros(new_capacity, dtype=np.int64)
            new_starts_buffer[:n_valid_starts] = starts_buffer[:n_valid_starts]
            starts_buffer = new_starts_buffer
            start_capacity = new_capacity

            # Also grow the interval buffer
            new_interval_capacity = int(interval_capacity * growth_factor)
            new_interval_buffer = np.zeros((new_interval_capacity, 2), dtype=np.int64)
            new_interval_buffer[:interval_capacity] = interval_buffer[
                :interval_capacity
            ]
            interval_buffer = new_interval_buffer
            interval_capacity = new_interval_capacity

        # Add the latest start to the buffer of valid starts
        starts_buffer[n_valid_starts] = latest_start
        n_valid_starts += 1

        # Set up intervals for cost evaluation
        current_end = current_obs_ind + 1

        # Fill the interval buffer with current valid starts and the current end
        interval_buffer[:n_valid_starts, 0] = starts_buffer[:n_valid_starts]
        interval_buffer[:n_valid_starts, 1] = current_end

        # Evaluate costs:
        cost_eval_t0 = time.perf_counter()
        agg_costs = np.sum(cost.evaluate(interval_buffer[:n_valid_starts]), axis=1)
        cost_eval_t1 = time.perf_counter()
        cost_eval_time += cost_eval_t1 - cost_eval_t0

        # Add the cost and penalty for a new segment (since last changepoint)
        # Reusing the agg_costs array to store the candidate optimal costs.
        agg_costs[:] += penalty + opt_cost[starts_buffer[:n_valid_starts]]
        candidate_opt_costs = agg_costs

        # Find the optimal cost and previous changepoint
        argmin_candidate_cost = np.argmin(candidate_opt_costs)
        min_start_idx = starts_buffer[argmin_candidate_cost]
        opt_cost[current_obs_ind + 1] = candidate_opt_costs[argmin_candidate_cost]
        prev_cpts[current_obs_ind] = min_start_idx

        # Pruning: update valid starts to exclude positions that cannot be optimal
        current_obs_ind_opt_cost = opt_cost[current_obs_ind + 1]
        abs_current_obs_opt_cost = np.abs(current_obs_ind_opt_cost)

        # Calculate pruning threshold with margin
        start_inclusion_threshold = (
            (
                current_obs_ind_opt_cost
                + abs_current_obs_opt_cost * (pruning_margin / 100.0)
            )
            + penalty  # Pruning inequality does not include added penalty.
            - split_cost  # Remove from right side of inequality.
        )

        # Apply pruning by filtering valid starts:
        valid_starts_mask = candidate_opt_costs <= start_inclusion_threshold
        n_new_valid_starts = np.sum(valid_starts_mask)
        starts_buffer[:n_new_valid_starts] = starts_buffer[:n_valid_starts][
            valid_starts_mask
        ]
        n_valid_starts = n_new_valid_starts

    return opt_cost[1:], get_changepoints(prev_cpts), cost_eval_time


def test_run_optimal_partitioning(
    cost: BaseCost, penalty: float, min_segment_length: int = 1
):
    cost.fit(changepoint_data)
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=True,
    )

    # Assert monotonicity of costs:
    if min_segment_length == 1:
        assert np.all(np.diff(pelt_result.optimal_costs) >= 0)
    assert len(pelt_result.changepoints) == n_segments - 1
    assert pelt_result.changepoints.tolist() == [seg_len]


def test_run_pelt(cost: BaseCost, penalty: float, min_segment_length=1):
    cost.fit(changepoint_data)
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert np.all(np.diff(pelt_result.optimal_costs) >= 0)
    assert len(pelt_result.changepoints) == n_segments - 1
    assert pelt_result.changepoints.tolist() == [seg_len]


def test_pelt_with_and_without_pruning_is_the_same(
    cost: BaseCost, penalty: float, min_segment_length: int = 1
):
    cost.fit(changepoint_data)

    opt_part_results = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )

    pelt_results = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    assert np.array_equal(opt_part_results.changepoints, pelt_results.changepoints)
    np.testing.assert_array_almost_equal(
        pelt_results.optimal_costs, opt_part_results.optimal_costs
    )


@pytest.mark.parametrize("min_segment_length", [1, 5, 10])
@pytest.mark.parametrize(
    "signal_end_index", list(range(20, len(alternating_sequence) + 1, 5))
)
def test_pelt_on_tricky_data(
    cost: BaseCost, penalty: float, min_segment_length: int, signal_end_index: int
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than 20.
    """
    # Original "run_pelt" found 7 changepoints.
    pruning_margin = 0.0
    cost.fit(alternating_sequence[0:signal_end_index])
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        pruning_margin=pruning_margin,
    )
    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=True,
    )

    assert np.array_equal(pelt_result.changepoints, opt_part_result.changepoints)
    np.testing.assert_almost_equal(
        pelt_result.optimal_costs[-1],
        pelt_partition_cost(
            alternating_sequence[0:signal_end_index],
            pelt_result.changepoints,
            cost,
            penalty=penalty,
        ),
        decimal=10,
        err_msg="PELT cost for final observation does not match partition cost.",
    )
    np.testing.assert_array_almost_equal(
        pelt_result.optimal_costs, opt_part_result.optimal_costs
    )


@pytest.mark.parametrize("min_segment_length", range(1, 20))
def test_pelt_min_segment_lengths(cost: BaseCost, penalty: float, min_segment_length):
    """
    Test PELT on a slightly more complex data set. There are
    change points every 20 samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 20.

    Segment length of 30 works again...
    """
    cost.fit(alternating_sequence)
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    cost.fit(alternating_sequence)
    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )

    assert np.array_equal(pelt_result.changepoints, opt_part_result.changepoints)
    np.testing.assert_array_almost_equal(
        pelt_result.optimal_costs, opt_part_result.optimal_costs, decimal=10
    )


@pytest.mark.parametrize("min_segment_length", range(31, 40))
def test_high_min_segment_length(cost: BaseCost, penalty: float, min_segment_length):
    """
    For all these segment lengths, the improved PELT implementation
    finds the same changepoints as the optimal partitioning.
    """
    cost.fit(alternating_sequence)
    # The PELT implementation (no longer) fails to find the same
    # changepoints as the optimal partitioning for these segment lengths,
    # when the segment length is greater than 30 and the pruning margin is zero.
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )

    assert np.array_equal(pelt_result.changepoints, opt_part_result.changepoints)
    np.testing.assert_array_almost_equal(
        pelt_result.optimal_costs, opt_part_result.optimal_costs
    )


@pytest.mark.parametrize("min_segment_length", [25])
def test_pelt_agrees_with_opt_part_longer_min_segment_length(
    cost: BaseCost, penalty: float, min_segment_length
):
    """
    For all these segment lengths, the PELT implementation
    fails to find the same changepoints as the optimal partitioning.
    """
    cost.fit(long_alternating_sequence)
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        pruning_margin=0.0,
    )

    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )

    assert np.array_equal(pelt_result.changepoints, opt_part_result.changepoints)
    np.testing.assert_array_almost_equal(
        pelt_result.optimal_costs, opt_part_result.optimal_costs
    )


@pytest.mark.parametrize("min_segment_length", [33, 39])
def test_comparing_skchange_to_ruptures_pelt_where_it_works(
    cost: BaseCost, penalty: float, min_segment_length: int
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every n samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 30.
    """
    cost.fit(long_alternating_sequence)

    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )
    opt_part_min_value = (
        evaluate_segmentation(cost, opt_part_result.changepoints)
        + (len(opt_part_result.changepoints)) * penalty
    )

    skchange_pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    skchange_pelt_min_value = (
        evaluate_segmentation(cost, skchange_pelt_result.changepoints)
        + (len(skchange_pelt_result.changepoints)) * penalty
    )

    rpt_model = rpt.Dynp(model="l2", min_size=min_segment_length, jump=1)
    rpt_model.fit(long_alternating_sequence)
    dyn_rpt_num_opt_part_cpts = np.array(
        rpt_model.predict(n_bkps=len(opt_part_result.changepoints))[:-1]
    )
    dyn_num_opt_part_cpts_min_value = (
        evaluate_segmentation(cost, dyn_rpt_num_opt_part_cpts)
        + (len(dyn_rpt_num_opt_part_cpts)) * penalty
    )

    ruptures_pelt_cpts = np.array(
        rpt.Pelt(model="l2", min_size=min_segment_length, jump=1).fit_predict(
            long_alternating_sequence, pen=penalty
        )[:-1]
    )
    rpt_pelt_min_value = (
        evaluate_segmentation(cost, ruptures_pelt_cpts)
        + (len(ruptures_pelt_cpts)) * penalty
    )
    assert np.array_equal(
        skchange_pelt_result.changepoints, opt_part_result.changepoints
    )
    assert np.array_equal(skchange_pelt_result.changepoints, dyn_rpt_num_opt_part_cpts)
    assert np.array_equal(skchange_pelt_result.changepoints, ruptures_pelt_cpts)

    assert skchange_pelt_min_value == opt_part_min_value
    assert skchange_pelt_min_value == dyn_num_opt_part_cpts_min_value
    assert skchange_pelt_min_value == rpt_pelt_min_value
    assert (
        np.abs(
            skchange_pelt_result.optimal_costs[-1] - opt_part_result.optimal_costs[-1]
        )
        < 1e-16
    )


@pytest.mark.parametrize("min_segment_length", [31, 32])
def test_compare_with_ruptures_pelt_where_restricted_pruning_works(
    cost: BaseCost, penalty: float, min_segment_length: int
):
    """
    Test PELT on a slightly more complex data set. There are
    change points every n samples, and the mean of the segments
    changes drastically. And the PELT implementation agrees with
    the optimal partitioning as long as the segment length is
    less than or equal to 30.
    """
    cost.fit(long_alternating_sequence)

    opt_part_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )
    opt_part_min_value = (
        evaluate_segmentation(cost, opt_part_result.changepoints)
        + (len(opt_part_result.changepoints)) * penalty
    )

    # Compare with 'improved PELT':
    skchange_pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )
    skchange_pelt_min_value = (
        evaluate_segmentation(cost, skchange_pelt_result.changepoints)
        + (len(skchange_pelt_result.changepoints)) * penalty
    )

    rpt_model = rpt.Dynp(model="l2", min_size=min_segment_length, jump=1)
    rpt_model.fit(long_alternating_sequence)
    dyn_rpt_num_opt_part_cpts = np.array(
        rpt_model.predict(n_bkps=len(opt_part_result.changepoints))[:-1]
    )
    dyn_num_opt_part_cpts_min_value = (
        evaluate_segmentation(cost, dyn_rpt_num_opt_part_cpts)
        + (len(dyn_rpt_num_opt_part_cpts)) * penalty
    )

    ruptures_pelt_cpts = np.array(
        rpt.Pelt(model="l2", min_size=min_segment_length, jump=1).fit_predict(
            long_alternating_sequence, pen=penalty
        )[:-1]
    )
    rpt_pelt_min_value = (
        evaluate_segmentation(cost, ruptures_pelt_cpts)
        + (len(ruptures_pelt_cpts)) * penalty
    )
    assert np.array_equal(
        skchange_pelt_result.changepoints, opt_part_result.changepoints
    )
    assert np.array_equal(skchange_pelt_result.changepoints, dyn_rpt_num_opt_part_cpts)

    assert skchange_pelt_min_value == opt_part_min_value
    assert skchange_pelt_min_value == dyn_num_opt_part_cpts_min_value
    assert skchange_pelt_min_value < rpt_pelt_min_value
    assert (
        np.abs(
            skchange_pelt_result.optimal_costs[-1] - opt_part_result.optimal_costs[-1]
        )
        < 1e-16
    )


@pytest.mark.parametrize("min_segment_length", [1, 2, 5, 10])
def test_pelt_dense_changepoints_parametrized(cost: BaseCost, min_segment_length):
    """
    Test PELT with penalty=0.0 to ensure we get changepoints as dense as possible
    allowed by min_segment_length, for different min_segment_length values.
    """
    increasing_data = np.linspace(0, 1 * seg_len, seg_len).reshape(-1, 1)
    penalty = 0.0
    cost.fit(increasing_data)
    pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=min_segment_length,
    )

    # Expected changepoints are at every min_segment_length interval
    expected_changepoints = [
        i * min_segment_length
        for i in range(1, len(increasing_data) // min_segment_length)
    ]

    assert np.array_equal(pelt_result.changepoints, expected_changepoints)


def test_invalid_costs():
    """
    Test that PELT raises an error when given an invalid cost argument.
    """
    with pytest.raises(ValueError, match="cost"):
        PELT(cost="l2")
    with pytest.raises(ValueError, match="cost"):
        PELT(cost=CUSUM())
    with pytest.raises(ValueError, match="cost"):
        cost = L2Cost()
        cost.set_tags(is_penalised=True)  # Simulate a penalised score
        PELT(cost=cost)


@pytest.mark.parametrize("step_size", [3, 5, 10])
def test_pelt_with_step_size(cost: BaseCost, penalty: float, step_size: int):
    """Test PELT with jump parameter enabled and min_segment_length > 2."""

    # Run PELT with step_size > 1:
    jump_pelt_model = PELT(
        cost=cost,
        step_size=step_size,
        penalty=penalty,
    )
    jump_pelt_model.fit(alternating_sequence)
    pelt_changepoints = jump_pelt_model.predict(alternating_sequence)[
        "ilocs"
    ].to_numpy()

    assert np.all(pelt_changepoints % step_size == 0)

    # Compare with ruptures implementation for validation
    rpt_model = rpt.Pelt(model="l2", min_size=step_size, jump=step_size)
    rpt_changepoints = np.array(
        rpt_model.fit_predict(alternating_sequence, pen=penalty)[:-1]
    )

    # Check if our implementation finds reasonable changepoints:
    assert len(pelt_changepoints) > 0

    assert len(pelt_changepoints) == len(rpt_changepoints)
    assert np.array_equal(pelt_changepoints, rpt_changepoints)

    # Test that the optimal cost at the last observation is correct:
    pelt_cost = evaluate_segmentation(jump_pelt_model.fitted_cost, pelt_changepoints)
    pelt_min_value = pelt_cost + (len(pelt_changepoints)) * penalty
    assert np.isclose(
        jump_pelt_model.scores.to_numpy()[-1], pelt_min_value, atol=1e-10
    ), (
        f"Expected PELT cost at the last observation to be {pelt_min_value}, "
        f"got {jump_pelt_model.scores.to_numpy()[-1]}"
    )


def test_jump_pelt_pruning_fraction(cost: BaseCost, penalty: float):
    """Test pruning fraction is zero when not pruning, and < 1.0 when pruning."""
    cost.fit(alternating_sequence)
    step_size = 5

    # Run with pruning disabled
    pelt_result_no_pruning = _run_pelt_with_step_size(
        cost=cost,
        penalty=penalty,
        step_size=step_size,
        prune=False,
    )

    # Run with pruning enabled
    pelt_result_with_pruning = _run_pelt_with_step_size(
        cost=cost,
        penalty=penalty,
        step_size=step_size,
    )

    # Check that pruning fraction is 1.0 when pruning is disabled
    assert pelt_result_no_pruning.pruning_fraction == 0.0, (
        f"Expected pruning fraction to be 0.0 when prune=False, "
        f"got {pelt_result_no_pruning.pruning_fraction}"
    )

    # Check that pruning fraction is less than 1.0 when pruning is enabled
    assert 0.0 < pelt_result_with_pruning.pruning_fraction < 1.0, (
        f"Expected pruning fraction to be between 0.0 and 1.0 when prune=True, "
        f"got {pelt_result_with_pruning.pruning_fraction}"
    )

    # Also check that the changepoints are the same regardless of pruning
    assert np.array_equal(
        pelt_result_no_pruning.changepoints, pelt_result_with_pruning.changepoints
    ), "Changepoints should be the same regardless of pruning"

    # Check that optimal costs are the same
    np.testing.assert_array_equal(
        pelt_result_no_pruning.optimal_costs, pelt_result_with_pruning.optimal_costs
    )


@pytest.mark.parametrize("min_segment_length", [30, 35, 40])
@pytest.mark.parametrize("common_penalty", [1.6120892290743671, 1.6562305936619783])
def test_old_pelt_failing_with_large_min_segment_length(
    common_penalty: float, min_segment_length: int
):
    """Test the CROPS algorithm for path solutions to penalized CPD.

    Reference: https://arxiv.org/pdf/1412.3617
    """
    cost = L2Cost()

    # Generate test data:
    dataset = generate_alternating_data(
        n_segments=2,
        segment_length=100,
        p=1,
        mean=3.0,
        variance=4.0,
        random_state=42,
    )

    # Fit the change point detector:
    cost.fit(dataset)

    # Check that the results are as expected:
    # Optimal start for the final point: Index 101
    opt_part_result = _run_pelt(
        cost=cost,
        penalty=common_penalty,
        min_segment_length=min_segment_length,
        prune=False,
    )

    improved_pelt_result = _run_pelt(
        cost=cost,
        penalty=common_penalty,
        min_segment_length=min_segment_length,
    )

    old_pelt_result = old_run_pelt(
        cost=cost,
        penalty=common_penalty,
        min_segment_length=min_segment_length,
    )

    # PELT objective values:
    opt_part_min_value = evaluate_segmentation(
        cost, opt_part_result.changepoints
    ) + common_penalty * len(opt_part_result.changepoints)

    improved_pelt_min_value = evaluate_segmentation(
        cost, improved_pelt_result.changepoints
    ) + common_penalty * len(improved_pelt_result.changepoints)

    old_pelt_min_value = evaluate_segmentation(
        cost, old_pelt_result.changepoints
    ) + common_penalty * len(old_pelt_result.changepoints)

    # Compare results:
    assert np.array_equal(
        opt_part_result.changepoints, improved_pelt_result.changepoints
    )
    # Old PELT result should be different from the optimal partitioning result:
    assert not np.array_equal(
        opt_part_result.changepoints, old_pelt_result.changepoints
    )

    np.testing.assert_array_equal(
        opt_part_result.optimal_costs, improved_pelt_result.optimal_costs
    )
    assert old_pelt_min_value > opt_part_min_value, (
        f"Expected old PELT cost to be greater than optimal partitioning cost, "
        f"got {old_pelt_min_value} vs {opt_part_min_value}"
    )
    assert improved_pelt_min_value == opt_part_min_value, (
        f"Expected improved PELT cost to be equal to optimal partitioning cost, "
        f"got {improved_pelt_min_value} vs {opt_part_min_value}"
    )


def test_pelt_with_fewer_samples_than_min_segment_length_throws():
    """Test that PELT raises when `n_samples` is less than `min_segment_length`."""
    cost = L2Cost()
    data = np.random.randn(5, 1)  # Less than min_segment_length of 10
    cost.fit(data)

    with pytest.raises(
        ValueError,
        match="The `min_segment_length` cannot be larger than the number of samples",
    ):
        _run_pelt(cost, penalty=1.0, min_segment_length=10)


def test_jump_pelt_with_fewer_samples_than_step_size_throws():
    """Test that PELT-step_size  raises when `n_samples` is less than `step_size`."""
    cost = L2Cost()
    data = np.random.randn(5, 1)  # Less than step_size of 10

    # Creating the PELT model with step_size=10
    jump_pelt_model = PELT(cost=cost, step_size=10, penalty=1.0)

    with pytest.raises(
        ValueError,
        match="The `step_size` cannot be larger than the number of samples",
    ):
        jump_pelt_model.fit(data)
        jump_pelt_model.predict(data)


def test_jump_pelt_with_fewer_samples_than_twice_step_size_returns_single_interval():
    """Test that PELT-jump_pelt raises when `n_samples` is less than `step_size`."""
    cost = L2Cost()
    data = np.random.randn(10, 1)  # Less than step_size of 10

    # Creating the PELT model with step_size=6
    jump_pelt_model = PELT(cost=cost, step_size=6, penalty=1.0)

    jump_pelt_model.fit(data)
    jump_pelt_res = jump_pelt_model.predict(data)
    # Expect PELT-step_size to return an empty array when n_samples < 2 * step_size.
    np.testing.assert_array_equal(jump_pelt_res["ilocs"].to_numpy(), np.array([]))


def test_pelt_min_segment_length_one_agrees_with_regular_run_pelt(
    cost: BaseCost, penalty: float
):
    """Test that PELT with min_segment_length=1 agrees with run_pelt."""
    cost.fit(alternating_sequence)

    regular_pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=1,
    )
    no_pruning_pelt_result = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=1,
        prune=False,
    )
    np.testing.assert_array_equal(
        regular_pelt_result.changepoints, no_pruning_pelt_result.changepoints
    )
    np.testing.assert_array_equal(
        regular_pelt_result.optimal_costs, no_pruning_pelt_result.optimal_costs
    )

    min_seg_length_one_pelt_result = _run_pelt_min_segment_length_one(
        cost,
        penalty=penalty,
    )
    no_pruning_min_seg_length_one_pelt_result = _run_pelt_min_segment_length_one(
        cost,
        penalty=penalty,
        prune=False,
    )
    assert no_pruning_min_seg_length_one_pelt_result.pruning_fraction == 0.0, (
        "Expected no pruning when min_segment_length=1 and prune=False, "
        f"got {no_pruning_min_seg_length_one_pelt_result.pruning_fraction}"
    )

    assert (
        min_seg_length_one_pelt_result == regular_pelt_result
    ), "Expected PELT with min_segment_length=1 to agree with regular PELT."
    assert no_pruning_min_seg_length_one_pelt_result == no_pruning_pelt_result, (
        "Expected PELT with min_segment_length=1 and prune=False to agree with "
        "regular PELT with prune=False."
    )


def test_pelt_min_segment_length_one_throws_if_zero_sample():
    """Test that PELT with min_segment_length=1 raises an error if no samples."""
    cost = L2Cost()
    data = np.empty((0, 1))  # No samples
    cost.fit(data)

    with pytest.raises(
        ValueError,
        match="The number of samples for the fitted cost must be at least one.",
    ):
        _run_pelt_min_segment_length_one(cost, penalty=1.0)


def test_constructing_PELTResult_with_differing_array_sizes_raises_error():
    """Test that PELTResult raises an error if arrays have different sizes."""
    with pytest.raises(ValueError, match="All input arrays must have the same length."):
        PELTResult.new(
            optimal_costs=np.array([1.0, 2.0]),
            previous_change_points=np.array([0]),
            pruning_fraction=0.0,
        )


def test_comparing_PELTResult_with_non_PELTResult_returns_false():
    """Test that PELTResult comparison with non-PELTResult returns False."""
    pelt_result = PELTResult.new(
        optimal_costs=np.array([1.0, 2.0]),
        previous_change_points=np.array([0, 1]),
        pruning_fraction=0.0,
    )
    assert (
        not pelt_result == "not a PELTResult object"
    ), "Expected comparison with non-PELTResult to return False."


def test_PELTResult_cannot_be_hashed():
    """Test that PELTResult cannot be used as a key in a dictionary."""
    pelt_result = PELTResult.new(
        optimal_costs=np.array([1.0, 2.0]),
        previous_change_points=np.array([0, 1]),
        pruning_fraction=0.0,
    )
    with pytest.raises(TypeError, match="unhashable type: 'PELTResult'"):
        _ = {pelt_result: "value"}  # Attempt to use PELTResult as a dict key


def test_PELT_step_size_less_than_min_segment_length():
    """Test that PELT with step_size < min_segment_length raises an error."""
    cost = L2Cost()
    data = np.random.randn(100, 1)  # Enough samples for testing
    cost.fit(data)

    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "PELT `min_segment_length`(=10) cannot be "
                "greater than the `step_size`(=5) > 1."
            )
        ),
    ):
        PELT(
            cost=cost,
            step_size=5,  # Jump step less than min_segment_length
            penalty=1.0,
            min_segment_length=10,  # Min segment length is larger
        ).fit_predict(data)


def test_run_pelt_with_pruning_margin_decreases_pruning_fraction(
    cost: BaseCost, penalty: float
):
    """Test that increasing pruning margin decreases pruning fraction."""
    cost.fit(alternating_sequence)

    # Run PELT with a small pruning margin
    pelt_result_no_margin = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=4,
        pruning_margin=0.0,  # Small margin
    )

    # Run PELT with a larger pruning margin
    pelt_result_large_margin = _run_pelt(
        cost,
        penalty=penalty,
        min_segment_length=4,
        pruning_margin=10.0,  # Larger margin
    )

    # Check that the pruning fraction decreases with larger margin
    assert (
        pelt_result_large_margin.pruning_fraction
        < pelt_result_no_margin.pruning_fraction
    ), (
        "Expected larger pruning margin to decrease pruning fraction, "
        f"got {pelt_result_large_margin.pruning_fraction} < "
        f"{pelt_result_no_margin.pruning_fraction}"
    )


def test_run_pelt_min_seglen_one_with_pruning_margin_decreases_pruning_fraction(
    cost: BaseCost, penalty: float
):
    """Test that increasing pruning margin decreases pruning fraction."""
    cost.fit(alternating_sequence)

    # Run PELT with a small pruning margin
    pelt_result_no_margin = _run_pelt_min_segment_length_one(
        cost,
        penalty=penalty,
        pruning_margin=0.0,  # Small margin
    )

    # Run PELT with a larger pruning margin
    pelt_result_large_margin = _run_pelt_min_segment_length_one(
        cost,
        penalty=penalty,
        pruning_margin=10.0,  # Larger margin
    )

    # Check that the pruning fraction decreases with larger margin
    assert (
        pelt_result_large_margin.pruning_fraction
        < pelt_result_no_margin.pruning_fraction
    ), (
        "Expected larger pruning margin to decrease pruning fraction, "
        f"got {pelt_result_large_margin.pruning_fraction} < "
        f"{pelt_result_no_margin.pruning_fraction}"
    )


def test_run_pelt_step_size_with_pruning_margin_decreases_pruning_fraction(
    cost: BaseCost, penalty: float
):
    """Test that increasing pruning margin decreases pruning fraction."""
    cost.fit(alternating_sequence)

    # Run PELT with a small pruning margin
    pelt_result_no_margin = _run_pelt_with_step_size(
        cost,
        step_size=5,
        penalty=penalty,
        pruning_margin=0.0,  # Small margin
    )

    # Run PELT with a larger pruning margin
    pelt_result_large_margin = _run_pelt_with_step_size(
        cost,
        step_size=5,
        penalty=penalty,
        pruning_margin=10.0,  # Larger margin
    )

    # Check that the pruning fraction decreases with larger margin
    assert (
        pelt_result_large_margin.pruning_fraction
        < pelt_result_no_margin.pruning_fraction
    ), (
        "Expected larger pruning margin to decrease pruning fraction, "
        f"got {pelt_result_large_margin.pruning_fraction} < "
        f"{pelt_result_no_margin.pruning_fraction}"
    )
