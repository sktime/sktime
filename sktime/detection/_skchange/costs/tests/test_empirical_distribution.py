import time

import numpy as np
import pytest
from scipy import stats

from sktime.detection._skchange.costs._empirical_distribution_cost import (
    EmpiricalDistributionCost,
    make_approximate_mle_edf_cost_quantile_points,
    make_cumulative_edf_cache,
    make_fixed_cdf_cost_quantile_weights,
    numba_approximate_mle_edf_cost_cached_edf,
    numba_fixed_cdf_cost_cached_edf,
    numpy_approximate_mle_edf_cost_cached_edf,
    numpy_fixed_cdf_cost_cached_edf,
)
from sktime.detection._skchange.utils.numba import njit, numba_available
from sktime.detection._skchange.utils.numba.general import compute_finite_difference_derivatives


@njit
def evaluate_edf_of_sorted_data(
    sorted_data: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate empirical distribution function (EDF) from sorted data.

    Parameters
    ----------
    sorted_data : np.ndarray
        The sorted data segment.
    values : np.ndarray
        The values at which to evaluate the EDF.

    Returns
    -------
    np.ndarray
        The EDF values at the specified points.
    """
    if len(sorted_data) < 2:
        raise ValueError("Data segment must contain at least two elements.")

    # Use searchsorted to find indices where each value would fit in the sorted data:
    # Effectively counts how many elements in `sorted_data` are less than
    # or equal to each value in `values`.
    indices = np.searchsorted(sorted_data, values, side="right")

    # Normalize the counts to get the EDF values:
    segment_edf_values = indices / len(sorted_data)

    return segment_edf_values


def evaluate_empirical_distribution_function(
    data: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the empirical distribution function (EDF) of a segment of data.

    Parameters
    ----------
    data : np.ndarray
        The data segment for which to compute the EDF.
    values : np.ndarray
        The values at which to evaluate the EDF.

    Returns
    -------
    np.ndarray
        The EDF values at the specified points.
    """
    sorted_data = np.sort(data)

    segment_edf_values = evaluate_edf_of_sorted_data(
        sorted_data,
        values,
    )

    return segment_edf_values


def direct_mle_edf_cost(
    xs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
) -> np.ndarray:
    """Compute exact empirical distribution cost.

    This function computes the empirical distribution cost for a sequence `xs` with
    given segment cuts defined by `segment_starts` and `segment_ends`. The cost is
    computed as the integrated log-likelihood of the empirical distribution function
    (EDF) for each segment. The EDF is evaluated at the sorted values of `xs`, excluding
    the first and last samples to avoid boundary effects.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.

    Returns
    -------
    np.ndarray
        1D array of empirical distribution costs for each segment defined by `cuts`.
    """
    assert xs.ndim == 1, "Input data must be a 1D array."
    edf_eval_points = np.sort(xs)[1:-1]  # Exclude the first and last samples
    n_samples = len(xs)
    reciprocal_full_data_cdf_weights = np.square(n_samples) / (
        np.arange(2, n_samples, dtype=np.float64)
        * np.arange(n_samples - 2, 0, -1, dtype=np.float64)
    )

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        segment_edf = evaluate_empirical_distribution_function(
            segment_data,
            edf_eval_points,
        )

        # Clip to avoid log(0) issues:
        segment_edf = np.clip(segment_edf, 1e-10, 1 - 1e-10)
        one_minus_segment_edf = 1.0 - segment_edf

        integrated_ll_at_mle = (segment_length / n_samples) * (
            np.sum(
                (
                    segment_edf * np.log(segment_edf)
                    + one_minus_segment_edf * np.log(one_minus_segment_edf)
                )
                * reciprocal_full_data_cdf_weights
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * integrated_ll_at_mle

    return segment_costs


def approximate_mle_edf_cost(
    xs: np.ndarray,
    quantile_points: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_costs: np.ndarray | None = None,
) -> np.ndarray:
    """Compute approximate empirical distribution cost.

    Using `num_quantiles` quantile values to approximate the empirical distribution cost
    for a sequence `xs` with given segment cuts. The cost is computed as the integrated
    log-likelihood of the empirical distribution function (EDF), approximated by
    evaluating the EDF at `num_quantiles` specific quantiles.

    Parameters
    ----------
    xs : np.ndarray
        The first sequence.
    cuts : np.ndarray
        The cut intervals to consider.
    num_quantiles : int
        The number of terms to use in the approximation.

    Returns
    -------
    np.ndarray
        1D array of empirical distribution costs for each segment defined by `cuts`.
    """
    assert xs.ndim == 1, "Input data must be a 1D array."
    n_samples = len(xs)
    num_quantiles = len(quantile_points)

    if num_quantiles <= 0:
        raise ValueError("num_quantiles must be a positive integer.")
    if num_quantiles > len(xs) - 2:
        raise ValueError(
            "num_quantiles should not be greater than the number of samples minus 2."
        )

    # Constant scaling term from the approximation:
    edf_integration_scale = -np.log(2 * n_samples - 1)

    # Initialize segment costs if not provided:
    if segment_costs is None:
        segment_costs = np.zeros(len(segment_starts), dtype=np.float64)

    segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)
    one_minus_segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)
    log_segment_edf_at_quantiles = np.zeros(num_quantiles, dtype=np.float64)

    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        if segment_length <= 0:
            raise ValueError("Invalid segment length.")

        segment_data = xs[segment_start:segment_end]
        segment_edf_at_quantiles = evaluate_empirical_distribution_function(
            segment_data,
            quantile_points,
        )

        # Clip to within (0, 1) to avoid log(0) issues:
        np.clip(segment_edf_at_quantiles, 1e-10, 1 - 1e-10, segment_edf_at_quantiles)
        one_minus_segment_edf_at_quantiles[:] = 1.0 - segment_edf_at_quantiles[:]

        ### Begin computing integrated log-likelihood for the segment ###
        segment_ll_at_mle = 0.0

        # Compute the first term: sum(F(t))*log(F(t))
        np.log(segment_edf_at_quantiles, log_segment_edf_at_quantiles)

        # Multiply together, storing in one_minus_segment_edf_at_quantiles:
        np.multiply(
            segment_edf_at_quantiles,
            log_segment_edf_at_quantiles,
            one_minus_segment_edf_at_quantiles,
        )
        segment_ll_at_mle += np.sum(one_minus_segment_edf_at_quantiles)

        # Compute the second term: sum(1 - F(t))*log(1 - F(t))
        one_minus_segment_edf_at_quantiles[:] = 1 - segment_edf_at_quantiles
        np.log(one_minus_segment_edf_at_quantiles, log_segment_edf_at_quantiles)

        # Multiply together, storing in segment_edf_at_quantiles:
        np.multiply(
            one_minus_segment_edf_at_quantiles,
            log_segment_edf_at_quantiles,
            segment_edf_at_quantiles,
        )
        segment_ll_at_mle += np.sum(segment_edf_at_quantiles)

        segment_ll_at_mle *= (
            -2.0 * edf_integration_scale / num_quantiles
        ) * segment_length
        ### Done computing integrated log-likelihood for the segment ###

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * segment_ll_at_mle

    return segment_costs


def fixed_cdf_empirical_distribution_cost(
    xs: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    fixed_quantiles: np.ndarray,
    fixed_ts: np.ndarray,
    quantile_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the empirical distribution cost on a refernce cdf.

    This function computes the empirical distribution cost for a sequence `xs` with
    given segment cuts defined by `segment_starts` and `segment_ends`. The cost is
    computed as the integrated log-likelihood of the empirical distribution function
    (EDF) for each segment, approximated by a sum over the provided quantiles.

    Parameters
    ----------
    xs : np.ndarray
        The input data array.
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.

    Returns
    -------
    np.ndarray
        1D array of empirical distribution cost evaluated for fixed cdf.
    """
    assert xs.ndim == 1, "Input data must be a 1D array."

    # Compute the integrals weights by approximating the derivative of the fixed CDF:
    if len(fixed_quantiles) < 3:
        raise ValueError("At least three fixed quantile values are required.")

    one_minus_fixed_quantiles = 1.0 - fixed_quantiles

    if quantile_weights is None:
        reciprocal_fixed_cdf_weights = 1.0 / (
            fixed_quantiles * one_minus_fixed_quantiles
        )
        fixed_quantile_derivatives = compute_finite_difference_derivatives(
            ts=fixed_ts,
            ys=fixed_quantiles,
        )
        quantile_weights = reciprocal_fixed_cdf_weights * fixed_quantile_derivatives

    segment_costs = np.zeros(len(segment_starts), dtype=np.float64)
    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        segment_length = segment_end - segment_start
        segment_data = xs[segment_start:segment_end]

        segment_edf_per_sample = evaluate_empirical_distribution_function(
            segment_data,
            fixed_ts,
        )

        # Clip to avoid log(0) issues:
        segment_edf_per_sample = np.clip(segment_edf_per_sample, 1e-10, 1 - 1e-10)
        one_minus_segment_empirical_distribution_per_sample = (
            1.0 - segment_edf_per_sample
        )

        integrated_ll_at_mle = segment_length * (
            np.sum(
                (
                    segment_edf_per_sample * np.log(fixed_quantiles)
                    + one_minus_segment_empirical_distribution_per_sample
                    * np.log(one_minus_fixed_quantiles)
                )
                * quantile_weights
            )
        )

        # The cost equals twice the negative integrated log-likelihood:
        segment_costs[i] = -2.0 * integrated_ll_at_mle

    return segment_costs


def test_fixed_cdf_empirical_distribution_cost_vs_direct_cost():
    xs = np.array([1, 5, 2, 3, 4])
    # xs = np.arange(100)
    segment_starts = np.array([0, 3])
    segment_ends = np.array([3, 5])
    segment_starts = np.array([0])
    segment_ends = np.array([len(xs)])

    # Direct cost:
    direct_cost = direct_mle_edf_cost(xs, segment_starts, segment_ends)

    # Fixed CDF cost:
    quantile_points = np.sort(xs)[1:-1]  # Exclude first and last samples
    # quantiles = np.searchsorted(xs, quantile_values, side="right") / len(xs)
    fixed_quantiles = np.arange(2, len(xs)) / len(xs)
    fixed_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=fixed_quantiles,
        fixed_ts=quantile_points,
    )

    fixed_cdf_quantile_weights = make_fixed_cdf_cost_quantile_weights(
        fixed_quantiles=fixed_quantiles,
        quantile_points=quantile_points,
    )

    fixed_cdf_cost_cached = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=fixed_quantiles,
        fixed_ts=quantile_points,
        quantile_weights=fixed_cdf_quantile_weights,
    )

    nudged_fixed_quantiles = fixed_quantiles + 0.05
    nudged_quantile_points = quantile_points + 0.5

    nudged_one_minus_fixed_quantiles = 1 - nudged_fixed_quantiles
    nudged_fixed_cdf_quantile_weights = make_fixed_cdf_cost_quantile_weights(
        fixed_quantiles=nudged_fixed_quantiles,
        quantile_points=nudged_quantile_points,
    )

    fixed_nudged_cdf_cost_cached = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=nudged_fixed_quantiles,
        fixed_ts=nudged_quantile_points,
        quantile_weights=nudged_fixed_cdf_quantile_weights,
    )

    fixed_cdf_cumulative_edf_cache = make_cumulative_edf_cache(
        xs, nudged_quantile_points
    )
    log_nudged_fixed_quantiles = np.log(nudged_fixed_quantiles)
    log_one_minus_fixed_nudged_quantiles = np.log(nudged_one_minus_fixed_quantiles)

    fixed_nudged_cdf_cost_cached_numba = numba_fixed_cdf_cost_cached_edf(
        fixed_cdf_cumulative_edf_cache,
        segment_starts=segment_starts,
        segment_ends=segment_ends,
        log_fixed_quantiles=log_nudged_fixed_quantiles,
        log_one_minus_fixed_quantiles=log_one_minus_fixed_nudged_quantiles,
        quantile_weights=nudged_fixed_cdf_quantile_weights,
    )

    fixed_nudged_cdf_cost_cached_numpy = numpy_fixed_cdf_cost_cached_edf(
        fixed_cdf_cumulative_edf_cache,
        segment_starts=segment_starts,
        segment_ends=segment_ends,
        log_fixed_quantiles=log_nudged_fixed_quantiles,
        log_one_minus_fixed_quantiles=log_one_minus_fixed_nudged_quantiles,
        quantile_weights=nudged_fixed_cdf_quantile_weights,
    )

    print(f"Direct cost: {direct_cost}")
    print(f"Fixed CDF cost: {fixed_cdf_cost}")
    print(f"Fixed CDF cost cached: {fixed_cdf_cost_cached}")

    print(f"Nudged fixed CDF cost cached: {fixed_nudged_cdf_cost_cached}")
    print(f"Nudged Fixed CDF cost cached v2: {fixed_nudged_cdf_cost_cached_numba}")

    np.testing.assert_equal(direct_cost, fixed_cdf_cost)
    np.testing.assert_equal(direct_cost, fixed_cdf_cost_cached)

    assert (
        direct_cost != fixed_nudged_cdf_cost_cached
    ), "Direct cost should not equal the nudged fixed CDF cost."
    np.testing.assert_equal(
        fixed_nudged_cdf_cost_cached, fixed_nudged_cdf_cost_cached_numba
    )
    np.testing.assert_equal(
        fixed_nudged_cdf_cost_cached, fixed_nudged_cdf_cost_cached_numpy
    )


def test_fixed_cdf_empirical_distribution_cost():
    # Test the standard normal CDF as fixed quantiles:
    quantiles = np.array([0.025, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.975])
    gaussian_quantile_ts = stats.norm.ppf(
        quantiles
    )  # Inverse CDF of the standard normal distribution

    gaussian_quantile_derivatives = stats.norm.pdf(gaussian_quantile_ts)
    approx_gaussian_quantile_derivatives = compute_finite_difference_derivatives(
        ts=gaussian_quantile_ts, ys=quantiles
    )
    np.testing.assert_allclose(
        gaussian_quantile_derivatives,
        approx_gaussian_quantile_derivatives,
        rtol=0.15,
        atol=1.0e-3,
    )

    denser_quantiles = np.exp(np.linspace(-3.0, 3.0, 100)) / (
        1.0 + np.exp(np.linspace(-3.0, 3.0, 100))
    )
    dense_quantile_ts = stats.norm.ppf(denser_quantiles)
    dense_quantile_derivatives = stats.norm.pdf(dense_quantile_ts)
    approx_dense_quantile_derivatives = compute_finite_difference_derivatives(
        ts=dense_quantile_ts, ys=denser_quantiles
    )

    # With denser sampling, much higher accuracy is expected:
    np.testing.assert_allclose(
        dense_quantile_derivatives,
        approx_dense_quantile_derivatives,
        rtol=1.0e-10,
        atol=1.0e-4,
    )

    xs = stats.norm.rvs(size=500, random_state=42)
    cdf_eval_ts = np.linspace(np.min(xs), np.max(xs), 100)

    segment_starts = np.array([0])
    segment_ends = np.array([500])

    empirical_quantiles = np.clip(
        evaluate_empirical_distribution_function(xs, cdf_eval_ts), 1e-10, 1 - 1e-10
    )
    gaussian_quantiles = stats.norm.cdf(cdf_eval_ts)
    laplace_quantiles = stats.laplace.cdf(cdf_eval_ts)

    fixed_empirical_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=empirical_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    fixed_gaussian_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=gaussian_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    # laplacian_quantile_ts = stats.laplace.ppf(quantiles)
    fixed_laplacian_cdf_cost = fixed_cdf_empirical_distribution_cost(
        xs,
        segment_starts,
        segment_ends,
        fixed_quantiles=laplace_quantiles,
        fixed_ts=cdf_eval_ts,
    )

    # NOTE: With the inverse-cdf weighting, the fixed laplacian CDF cost is lower.
    #       Without the weighting, we get as expected:
    #       empirical_cdf_cost < gaussian_cdf_cost < laplacian_cdf_cost.
    print(f"Fixed empirical CDF cost: {fixed_empirical_cdf_cost}")
    print(f"Fixed Gaussian CDF cost: {fixed_gaussian_cdf_cost}")
    print(f"Fixed Laplacian CDF cost: {fixed_laplacian_cdf_cost}")
    assert np.all(fixed_laplacian_cdf_cost < fixed_gaussian_cdf_cost), (
        "Laplacian CDF cost should be lower than Gaussian CDF cost,"
        " on standard normal data, with inverse-cdf weighting."
    )
    assert np.all(fixed_gaussian_cdf_cost < fixed_empirical_cdf_cost), (
        "Gaussian CDF cost should be lower than empirical CDF cost,"
        " on standard normal data, with inverse-cdf weighting."
    )


def test_evaluate_empirical_distribution_function():
    xs = np.array([1, 2, 3, 4, 5])

    edf_eval_points_1 = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    xs_edf_eval_point_quantiles = evaluate_empirical_distribution_function(
        xs, edf_eval_points_1
    )
    np.testing.assert_allclose(
        xs_edf_eval_point_quantiles,
        np.array([0.2, 0.4, 0.4, 0.6, 0.6]),
    )

    edf_eval_points_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xs_edf_eval_point_quantiles_equal = evaluate_empirical_distribution_function(
        xs, edf_eval_points_2
    )
    np.testing.assert_allclose(
        xs_edf_eval_point_quantiles_equal,
        np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
    )

    # Add more tests with different data
    xs2 = np.array([1, 1, 2, 2, 3, 3])
    edf_eval_points2 = np.array([0.99, 1.99, 2.99])
    xs2_edf_eval_point_quantiles = evaluate_empirical_distribution_function(
        xs2, edf_eval_points2
    )
    np.testing.assert_allclose(
        xs2_edf_eval_point_quantiles,
        np.array([0.0, 1 / 3, 2 / 3]),
    )


def test_evaluate_edf_from_cache():
    xs = np.array([1, 2, 3, 4, 5])

    edf_eval_points_1 = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    approx_cost_cache = make_cumulative_edf_cache(xs, edf_eval_points_1)
    xs_edf_eval_point_quantiles = (
        approx_cost_cache[-1, :] - approx_cost_cache[0, :]
    ) / len(xs)
    np.testing.assert_array_equal(
        xs_edf_eval_point_quantiles,
        np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
    )

    edf_eval_points_2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    approx_cost_cache_2 = make_cumulative_edf_cache(xs, edf_eval_points_2)
    xs_edf_eval_point_quantiles_2 = (
        approx_cost_cache_2[-1, :] - approx_cost_cache_2[0, :]
    ) / len(xs)
    np.testing.assert_array_equal(
        xs_edf_eval_point_quantiles_2,
        np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    )


@pytest.mark.parametrize(
    ["tolerance", "n_samples"], [(0.10, 60), (0.05, 120), (0.025, 1000)]
)
def test_approximate_vs_direct_cost_on_longer_data(tolerance: float, n_samples: int):
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(
        size=n_samples, loc=5
    )  # Shifted mean for the second segment
    xs = np.concatenate([first_segment, second_segment]).reshape(-1, 1)

    correct_segment_starts = np.array([0, n_samples])
    correct_segment_ends = np.array([n_samples, len(xs)])
    no_change_segment_starts = np.array([0])
    no_change_segment_ends = np.array([len(xs)])

    # Suggested value based on the length of xs:
    num_quantiles = int(4 * np.log(len(xs)))

    approx_quantile_points, _ = make_approximate_mle_edf_cost_quantile_points(
        xs, num_quantiles
    )

    one_change_approx_costs = approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=correct_segment_starts,
        segment_ends=correct_segment_ends,
    )
    one_change_direct_costs = direct_mle_edf_cost(
        xs[:, 0], correct_segment_starts, correct_segment_ends
    )

    relative_differences = np.abs(
        (one_change_approx_costs - one_change_direct_costs) / one_change_direct_costs
    )
    assert np.all(
        relative_differences < tolerance
    ), f"Relative differences exceed {tolerance * 100}%: {relative_differences}"

    single_segment_approx_cost = approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=no_change_segment_starts,
        segment_ends=no_change_segment_ends,
    )
    single_segment_direct_cost = direct_mle_edf_cost(
        xs[:, 0], no_change_segment_starts, no_change_segment_ends
    )
    single_segment_relative_difference = np.abs(
        (single_segment_approx_cost[0] - single_segment_direct_cost[0])
        / single_segment_direct_cost[0]
    )
    assert single_segment_relative_difference < tolerance, (
        f"Relative difference for single segment exceeds {tolerance * 100}%: "
        f"{single_segment_relative_difference}"
    )

    assert (
        single_segment_approx_cost - np.sum(one_change_approx_costs) > 0
    ), "Approximate cost for no change should be greater than for two segments."
    assert (
        single_segment_direct_cost - np.sum(one_change_direct_costs) > 0
    ), "Direct cost for no change should be greater than for two segments."


@pytest.mark.parametrize(
    "cached_approx_cost_func",
    [
        numba_approximate_mle_edf_cost_cached_edf,
        numpy_approximate_mle_edf_cost_cached_edf,
    ],
)
@pytest.mark.parametrize(
    ["rel_tol", "n_samples"], [(5.0e-2, 60), (1.0e-3, 120), (1.0e-4, 1600)]
)
def test_approximate_vs_precached_approximate_cost(
    cached_approx_cost_func, rel_tol: float, n_samples: int
):
    np.random.seed(42)  # For reproducibility
    first_segment = np.random.normal(size=n_samples)
    second_segment = np.random.normal(size=n_samples, loc=5)
    xs = np.concatenate([first_segment, second_segment]).reshape(-1, 1)
    num_quantiles = int(4 * np.log(len(xs)))

    approx_quantile_points, _ = make_approximate_mle_edf_cost_quantile_points(
        xs, num_quantiles
    )

    correct_cuts = np.array([[0, n_samples], [n_samples, len(xs)]])
    correct_segment_starts = correct_cuts[:, 0]
    correct_segment_ends = correct_cuts[:, 1]

    no_change_cuts = np.array([[0, len(xs)]])
    no_change_segment_starts = no_change_cuts[:, 0]
    no_change_segment_ends = no_change_cuts[:, 1]

    # Compare approximate vs precached on correct cuts
    approx_costs = approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=correct_segment_starts,
        segment_ends=correct_segment_ends,
    )
    approx_cost_edf_cache = make_cumulative_edf_cache(xs, approx_quantile_points)
    precached_costs = cached_approx_cost_func(
        approx_cost_edf_cache, correct_segment_starts, correct_segment_ends
    )
    np.testing.assert_allclose(approx_costs, precached_costs, rtol=rel_tol)

    # Compare on single segment
    single_approx_cost = approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=no_change_segment_starts,
        segment_ends=no_change_segment_ends,
    )
    single_precached_cost = cached_approx_cost_func(
        approx_cost_edf_cache, no_change_segment_starts, no_change_segment_ends
    )

    np.testing.assert_allclose(single_approx_cost, single_precached_cost, rtol=rel_tol)


def test_direct_vs_approximation_runtime(n_samples=10_000):
    xs = np.random.normal(size=n_samples).reshape(-1, 1)
    per_hundred_step_cuts = np.array(
        [[i * 100, (i + 1) * 100] for i in range(len(xs) // 100)]
    )
    per_hundred_step_segment_starts = per_hundred_step_cuts[:, 0]
    per_hundred_step_segment_ends = per_hundred_step_cuts[:, 1]
    num_approx_quantiles = int(4 * np.log(n_samples))

    approx_quantile_points, _ = make_approximate_mle_edf_cost_quantile_points(
        xs, num_approx_quantiles
    )

    # - Call once in case of JIT compilation overhead:
    direct_cost = direct_mle_edf_cost(
        xs[:, 0], per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    start_time = time.perf_counter()
    direct_cost = direct_mle_edf_cost(
        xs[:, 0], per_hundred_step_segment_starts, per_hundred_step_segment_ends
    )
    end_time = time.perf_counter()
    direct_cost_eval_time = end_time - start_time
    total_direct_cost = np.sum(direct_cost)

    assert (
        direct_cost_eval_time < 6.0e-2
    ), "Direct evaluation time should be less than 0.06 seconds."

    # Approximate evaluation:
    # - Call once in case of JIT compilation overhead:
    approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=per_hundred_step_segment_starts,
        segment_ends=per_hundred_step_segment_ends,
    )
    start_time = time.perf_counter()
    approx_cost = approximate_mle_edf_cost(
        xs[:, 0],
        quantile_points=approx_quantile_points[:, 0],
        segment_starts=per_hundred_step_segment_starts,
        segment_ends=per_hundred_step_segment_ends,
    )
    end_time = time.perf_counter()
    approximate_cost_eval_time = end_time - start_time
    total_approx_cost = np.sum(approx_cost)

    assert (
        approximate_cost_eval_time < 1.0e-2
    ), "Approximate evaluation time should be less than 0.01 sec."

    # Pre-caching the approximation:
    # - Call once in case of JIT compilation overhead:
    approx_cost_edf_cache = make_cumulative_edf_cache(
        xs, quantile_points=approx_quantile_points
    )
    numba_approximate_mle_edf_cost_cached_edf(
        approx_cost_edf_cache,
        per_hundred_step_segment_starts,
        per_hundred_step_segment_ends,
    )

    cache_start_time = time.perf_counter()
    approx_cost_edf_cache = make_cumulative_edf_cache(
        xs, quantile_points=approx_quantile_points
    )
    cache_end_time = time.perf_counter()
    cache_creation_time = cache_end_time - cache_start_time

    if numba_available:
        start_time = time.perf_counter()
        pre_cached_cost = numba_approximate_mle_edf_cost_cached_edf(
            approx_cost_edf_cache,
            per_hundred_step_segment_starts,
            per_hundred_step_segment_ends,
        )
        end_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        pre_cached_cost = numpy_approximate_mle_edf_cost_cached_edf(
            approx_cost_edf_cache,
            per_hundred_step_segment_starts,
            per_hundred_step_segment_ends,
        )
        end_time = time.perf_counter()

    pre_cached_eval_time = end_time - start_time
    total_pre_cached_cost = np.sum(pre_cached_cost)

    if numba_available:
        max_cache_creation_time = 5.0e-2
        max_pre_cached_eval_time = 1.0e-3
    else:
        max_cache_creation_time = 5.0e-1
        max_pre_cached_eval_time = 5.0e-2

    assert (
        cache_creation_time < max_cache_creation_time
    ), f"Cache creation should take less than {max_cache_creation_time:.2e} seconds."
    assert (
        pre_cached_eval_time < max_pre_cached_eval_time
    ), f"Pre-cached eval. should take less than {max_pre_cached_eval_time:.2e} sec."

    assert np.isclose(
        total_pre_cached_cost, total_approx_cost, rtol=1.0e-4
    ), "Pre-cached approximate cost does not match approximate cost within tolerance."
    assert np.isclose(
        total_direct_cost, total_pre_cached_cost, rtol=5.0e-2
    ), "Approximate cost does not match direct cost within tolerance."


def test_make_fixed_cdf_cost_quantile_weights_raises_value_error():
    """Test raises ValueError on insufficient quantiles."""
    fixed_quantiles = np.array([0.1, 0.5])  # Fewer than 3 quantiles
    quantile_points = np.array([1.0, 2.0])  # Corresponding points

    with pytest.raises(ValueError, match="At least three fixed quantile values"):
        make_fixed_cdf_cost_quantile_weights(fixed_quantiles, quantile_points)


def test_empirical_distribution_cost_default_num_quantiles():
    """Test EmpiricalDistributionCost with default number of approximation quantiles."""
    n_samples = 100
    X = np.random.normal(size=(n_samples, 1))  # Generate random data

    # Initialize EmpiricalDistributionCost with default parameters
    cost = EmpiricalDistributionCost(param=None, num_approximation_quantiles=None)
    assert cost.min_size == 10, "Default min_size should be 10 when not fitted."
    cost.fit(X)

    # Verify the default number of approximation quantiles
    expected_num_quantiles = int(np.ceil(4 * np.log(n_samples)))
    assert (
        cost.min_size == expected_num_quantiles
    ), f"Expected min_size to be {expected_num_quantiles}, but got {cost.min_size}."
    assert (
        cost.num_quantiles_ == expected_num_quantiles
    ), f"Expected {expected_num_quantiles} quantiles, but got {cost.num_quantiles_}."


def test_min_size_is_num_approximation_quantiles():
    """Test that min_size is set to num_approximation_quantiles."""
    num_approximation_quantiles = 20
    cost = EmpiricalDistributionCost(
        param=None, num_approximation_quantiles=num_approximation_quantiles
    )
    assert (
        cost.min_size == num_approximation_quantiles
    ), f"Expected min_size to be {num_approximation_quantiles}, but got {cost.min_size}"


def test_empirical_distribution_cost_check_fixed_param_errors():
    """Test all error cases for EmpiricalDistributionCost._check_fixed_param."""
    cost = EmpiricalDistributionCost()

    X = np.random.normal(size=(100, 2))  # Example input data

    # Case 1: Fixed samples and quantiles have mismatched shapes
    fixed_samples = np.array([1, 2, 3])
    fixed_quantiles = np.array([0.1, 0.2])  # Mismatched shape
    with pytest.raises(
        ValueError, match="samples must have a corresponding fixed quantile"
    ):
        cost._check_fixed_param((fixed_samples, fixed_quantiles), X)

    # Case 2: Fixed samples are not sorted and strictly increasing
    fixed_samples = np.array([3, 2, 1])  # Not sorted
    fixed_quantiles = np.array([0.1, 0.2, 0.3])
    with pytest.raises(
        ValueError, match="Fixed samples must be sorted, and strictly increasing."
    ):
        cost._check_fixed_param((fixed_samples, fixed_quantiles), X)

    # Case 3: Fixed quantiles are not sorted and strictly increasing
    fixed_samples = np.array([1, 2, 3])
    fixed_quantiles = np.array([0.3, 0.2, 0.1])  # Not sorted
    with pytest.raises(
        ValueError, match="Fixed CDF quantiles must be sorted, and strictly increasing."
    ):
        cost._check_fixed_param((fixed_samples, fixed_quantiles), X)

    # Case 4: Fixed quantiles are not within the interval [0, 1]
    fixed_samples = np.array([1, 2, 3])
    fixed_quantiles = np.array([-0.1, 0.5, 1.1])  # Out of bounds
    with pytest.raises(
        ValueError,
        match="Fixed quantiles must be within the closed interval \\[0, 1\\]",
    ):
        cost._check_fixed_param((fixed_samples, fixed_quantiles), X)

    # Case 5: Fixed samples and quantiles are 1D but X has multiple columns
    fixed_samples = np.array([1, 2, 3])
    fixed_quantiles = np.array([0.1, 0.5, 0.9])
    X = np.random.normal(size=(100, 3))  # Multi-column data
    result_samples, result_quantiles = cost._check_fixed_param(
        (fixed_samples, fixed_quantiles), X
    )
    assert result_samples.shape == (
        3,
        3,
    ), "Fixed samples should be tiled to match the number of columns in X."
    assert result_quantiles.shape == (
        3,
        3,
    ), "Fixed quantiles should be tiled to match the number of columns in X."


def test_empirical_distribution_cost_get_model_size_raises():
    """Test get_model_size raises when cost not fitted, no quantiles specified."""
    cost = EmpiricalDistributionCost(param=None, num_approximation_quantiles=None)
    with pytest.raises(
        ValueError,
        match="The cost is not fitted, and no number of quantiles was specified",
    ):
        cost.get_model_size(p=1)


def test_empirical_distribution_cost_check_fixed_param_ndim_error():
    """Test _check_fixed_param raises, fixed_cdf_quantiles or fixed_samples ndim > 2."""
    cost = EmpiricalDistributionCost()
    X = np.random.normal(size=(100, 2))  # Example input data

    # Case 1: fixed_samples has ndim > 2
    fixed_samples = np.random.normal(size=(3, 2, 2))  # Invalid shape
    fixed_quantiles = np.random.normal(size=(3, 2, 2))  # Invalid shape
    # fixed_quantiles = np.array([0.1, 0.5, 0.9])
    with pytest.raises(
        ValueError,
        match="Fixed samples and quantiles must be 1D or 2D arrays",
    ):
        cost._check_fixed_param((fixed_samples, fixed_quantiles), X)
