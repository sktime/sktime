import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from sktime.detection._skchange.change_detectors import PELT
from sktime.detection._skchange.costs._poisson_cost import (
    PoissonCost,
    poisson_log_likelihood,
    poisson_mle_rate_log_likelihood,
)


def test_poisson_log_likelihood():
    """Test that fast_poisson_log_likelihood agrees with scipy implementation."""
    # Generate random Poisson samples with known rate
    rate = 3.5
    poisson_sample = stats.poisson.rvs(rate, size=1000)

    # Calculate log-likelihood using our fast implementation
    fast_ll = poisson_log_likelihood(rate, poisson_sample)

    # Calculate log-likelihood using scipy
    scipy_ll = stats.poisson.logpmf(poisson_sample, rate).sum()

    # Check that they are close
    assert np.isclose(fast_ll, scipy_ll, rtol=1e-10)


def test_poisson_mle_log_likelihood():
    """Test that log likelihood evaluation agrees with scipy implementation."""
    # Generate random Poisson samples
    np.random.seed(42)
    true_rate = 5.0
    poisson_sample = stats.poisson.rvs(true_rate, size=1000)

    # Calculate MLE rate (sample mean)
    mle_rate = np.mean(poisson_sample)

    # Calculate log-likelihood using our fast implementation with MLE rate
    fast_ll = poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)

    # Calculate log-likelihood using scipy with MLE rate
    scipy_ll = stats.poisson.logpmf(poisson_sample, mle_rate).sum()

    # Check that they are close
    assert np.isclose(fast_ll, scipy_ll, rtol=1e-10)


def test_poisson_cost_fixed_param():
    """Test PoissonCost with fixed parameter."""
    # Generate Poisson data
    np.random.seed(42)
    rate = 4.0
    n_samples = 100
    poisson_data = stats.poisson.rvs(rate, size=(n_samples, 1))

    # Create PoissonCost with fixed rate
    cost = PoissonCost(param=rate)
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_cost = -2 * stats.poisson.logpmf(poisson_data, rate).sum()

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_poisson_cost_mle_param():
    """Test PoissonCost with MLE parameter."""
    # Generate Poisson data
    np.random.seed(42)
    true_rate = 4.0
    n_samples = 100
    poisson_data = stats.poisson.rvs(true_rate, size=(n_samples, 1))

    # Create PoissonCost with MLE rate
    cost = PoissonCost()
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate MLE rate
    mle_rate = np.mean(poisson_data)

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_cost = -2 * stats.poisson.logpmf(poisson_data, mle_rate).sum()

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_poisson_cost_input_validation():
    """Test input validation in PoissonCost."""
    # Try to fit with non-integer data
    with pytest.raises(ValueError):
        cost = PoissonCost()
        cost.fit(np.array([[1.5, 2.3], [3.1, 4.2]]))

    # Try to fit with negative integer data
    with pytest.raises(ValueError):
        cost = PoissonCost()
        cost.fit(np.array([[1, 2], [-3, 4]], dtype=int))

    # Try to create with negative rate parameter
    with pytest.raises(ValueError):
        cost = PoissonCost(param=-1.0)
        cost.fit(np.array([[1, 2], [3, 4]], dtype=int))


def test_poisson_cost_on_two_columns():
    """Test PoissonCost on two columns."""
    # Generate Poisson data
    np.random.seed(42)
    rate1 = 4.0
    rate2 = 2.0
    n_samples = 100
    poisson_data = np.column_stack(
        (
            stats.poisson.rvs(rate1, size=n_samples),
            stats.poisson.rvs(rate2, size=n_samples),
        )
    )

    # Create PoissonCost with fixed rate
    cost = PoissonCost(param=[rate1, rate2])
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_costs = [
        -2.0 * stats.poisson.logpmf(poisson_data[:, 0], rate1).sum(),
        -2.0 * stats.poisson.logpmf(poisson_data[:, 1], rate2).sum(),
    ]

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_costs[0], rtol=1e-10)
    assert np.isclose(costs[0, 1], expected_costs[1], rtol=1e-10)


def test_pelt_with_poisson_cost():
    """Test that PELT with PoissonCost correctly detects changes in Poisson data."""
    import scipy.stats as stats

    # Generate Poisson data with a change in rate
    np.random.seed(42)
    n_samples_per_segment = 100
    rate_before = 2.0
    rate_after = 5.0

    # First segment with rate_before
    segment1 = stats.poisson.rvs(rate_before, size=n_samples_per_segment)
    # Second segment with rate_after
    segment2 = stats.poisson.rvs(rate_after, size=n_samples_per_segment)

    # Combine segments
    data = np.concatenate([segment1, segment2])
    df = pd.DataFrame(data, columns=["count"])

    # True changepoint location
    true_cp = n_samples_per_segment

    # Apply PELT with PoissonCost
    cost = PoissonCost()
    detector = PELT(cost=cost)

    # Detect changepoints
    result = detector.fit_predict(df)

    # Check if detected changepoint is close to true changepoint
    detected_cp = result.iloc[0, 0] if len(result) == 1 else None

    # Allow a small margin of error (e.g., ±1 sample(s))
    margin = 1
    assert detected_cp is not None, "No changepoint detected"
    assert abs(detected_cp - true_cp) <= margin, (
        f"Detected CP at {detected_cp}, expected near {true_cp}"
    )

    # Test with multiple changepoints
    rate_third = 1.0
    segment3 = stats.poisson.rvs(rate_third, size=n_samples_per_segment)

    # Combine three segments
    multi_data = np.concatenate([segment1, segment2, segment3])
    multi_df = pd.DataFrame(multi_data, columns=["count"])

    # True changepoint locations
    true_cps = [n_samples_per_segment, 2 * n_samples_per_segment]

    # Apply PELT with PoissonCost
    multi_detector = PELT(cost=cost)
    multi_result = multi_detector.fit_predict(multi_df)

    # Check if detected changepoints are close to true changepoints
    assert len(multi_result) == 2, f"Expected 2 changepoints, got {len(multi_result)}"

    detected_cps = sorted(multi_result.values)
    for i, (detected, true) in enumerate(zip(detected_cps, true_cps)):
        assert abs(detected[0] - true) <= margin, (
            f"CP {i + 1}: detected at {detected}, expected near {true}"
        )


def test_poisson_cost_with_all_zeroes():
    """Test PoissonCost with an interval containing only zeroes."""
    # Create data with all zeroes
    n_samples = 100
    zero_data = np.zeros((n_samples, 1), dtype=int)

    # Test with MLE parameter (which should be 0 for all zeroes)
    cost_mle = PoissonCost()
    cost_mle.fit(zero_data)
    costs_mle = cost_mle.evaluate(np.array([[0, n_samples]]))

    # For all zeroes, the MLE rate is 0, and the log-likelihood should be 0
    # (as per implementation in poisson_cost_mle_params)
    assert costs_mle[0, 0] == 0.0

    # Test with fixed parameter
    fixed_rate = 0.001  # Small positive value to avoid division by zero
    cost_fixed = PoissonCost(param=fixed_rate)
    cost_fixed.fit(zero_data)
    costs_fixed = cost_fixed.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy
    expected_cost = -2 * stats.poisson.logpmf(0, fixed_rate).sum() * n_samples
    assert np.isclose(costs_fixed[0, 0], expected_cost, rtol=1e-10)


def test_poisson_cost_with_zeroes_and_ones():
    """Test PoissonCost with an interval containing only zeroes and ones."""
    # Create data with zeroes and ones
    np.random.seed(42)
    n_samples = 100
    data = np.random.choice([0, 1], size=(n_samples, 1))

    # Test with MLE parameter
    cost_mle = PoissonCost()
    cost_mle.fit(data)
    costs_mle = cost_mle.evaluate(np.array([[0, n_samples]]))

    # Calculate MLE rate and expected cost
    mle_rate = np.mean(data)
    expected_cost_mle = -2 * stats.poisson.logpmf(data, mle_rate).sum()
    assert np.isclose(costs_mle[0, 0], expected_cost_mle, rtol=1e-10)

    # Test with fixed parameter
    fixed_rate = 0.5
    cost_fixed = PoissonCost(param=fixed_rate)
    cost_fixed.fit(data)
    costs_fixed = cost_fixed.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy
    expected_cost_fixed = -2 * stats.poisson.logpmf(data, fixed_rate).sum()
    assert np.isclose(costs_fixed[0, 0], expected_cost_fixed, rtol=1e-10)

    # Test multiple columns with different distributions of zeroes and ones
    multi_data = np.column_stack(
        [
            np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # Mostly zeroes
            np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),  # Mostly ones
        ]
    )

    # Test with MLE parameter for multiple columns
    cost_multi = PoissonCost()
    cost_multi.fit(multi_data)
    costs_multi = cost_multi.evaluate(np.array([[0, n_samples]]))

    # Calculate expected costs
    expected_costs = [
        -2 * stats.poisson.logpmf(multi_data[:, i], np.mean(multi_data[:, i])).sum()
        for i in range(multi_data.shape[1])
    ]

    # Check each column's cost
    for i in range(multi_data.shape[1]):
        assert np.isclose(costs_multi[0, i], expected_costs[i], rtol=1e-10)
