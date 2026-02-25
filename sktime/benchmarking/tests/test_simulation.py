# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TimeSeriesSimulator."""

__author__ = ["IndarKumar"]

import numpy as np
import pandas as pd
import pytest

from sktime.benchmarking.simulation import TimeSeriesSimulator
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesSimulator),
    reason="Run tests for TimeSeriesSimulator only if explicitly requested",
)
class TestTimeSeriesSimulator:
    """Tests for TimeSeriesSimulator class."""

    def test_simulator_default(self):
        """Test simulator with default parameters."""
        sim = TimeSeriesSimulator(random_state=42)
        y = sim.simulate()

        assert isinstance(y, pd.Series)
        assert len(y) == 100
        assert isinstance(y.index, pd.DatetimeIndex)
        assert y.name == "simulated"

    def test_simulator_normal_distribution(self):
        """Test normal distribution generation."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 10, "scale": 2},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        # Check mean is roughly correct (within 3 sigma of expected)
        assert 8 < y.mean() < 12

    def test_simulator_poisson_distribution(self):
        """Test Poisson distribution generation."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="poisson",
            dist_params={"lam": 5},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 100
        # Poisson values should be non-negative integers
        assert (y >= 0).all()

    def test_simulator_with_linear_trend(self):
        """Test time series with linear trend."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 0, "scale": 1},
            trend="linear",
            trend_params={"slope": 1.0, "intercept": 0},
            random_state=42,
        )
        y = sim.simulate()

        # Check that there's an upward trend
        first_half_mean = y.iloc[:25].mean()
        second_half_mean = y.iloc[25:].mean()
        assert second_half_mean > first_half_mean

    def test_simulator_with_quadratic_trend(self):
        """Test time series with quadratic trend."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            trend="quadratic",
            trend_params={"a": 0.01, "b": 0, "c": 0},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 100
        # Quadratic trend should show acceleration
        first_quarter = y.iloc[:25].mean()
        last_quarter = y.iloc[75:].mean()
        assert last_quarter > first_quarter

    def test_simulator_with_exponential_trend(self):
        """Test time series with exponential trend."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            trend="exponential",
            trend_params={"base": 1.05, "scale": 1.0},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50

    def test_simulator_with_seasonality(self):
        """Test time series with seasonal component."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            seasonality=10,
            seasonality_strength=5.0,
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 100
        # Seasonal pattern should show variance
        assert y.std() > 0

    def test_simulator_with_custom_distribution(self):
        """Test custom distribution function."""

        def custom_dist(size, random_state):
            return random_state.beta(2, 5, size=size)

        sim = TimeSeriesSimulator(length=50, distribution=custom_dist, random_state=42)
        y = sim.simulate()

        assert len(y) == 50
        # Beta(2,5) values should be between 0 and 1
        assert (y >= 0).all() and (y <= 1).all()

    def test_simulator_with_custom_trend(self):
        """Test custom trend function."""

        def custom_trend(t):
            return 0.5 * t**2

        sim = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 0, "scale": 1},
            trend=custom_trend,
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50

    def test_simulator_with_noise(self):
        """Test adding Gaussian noise."""
        # Without noise
        sim1 = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 10, "scale": 1},
            noise_std=0.0,
            random_state=42,
        )
        y1 = sim1.simulate()

        # With noise - use different random state to get different base values
        sim2 = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 10, "scale": 1},
            noise_std=5.0,
            random_state=43,
        )
        y2 = sim2.simulate()

        # Both should produce valid series
        assert len(y1) == 50
        assert len(y2) == 50

    def test_simulator_multiple_seasonalities(self):
        """Test multiple seasonal components."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            seasonality=[7, 30],
            seasonality_strength=[2.0, 3.0],
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 100

    def test_simulator_single_strength_multiple_periods(self):
        """Test single seasonality_strength with multiple periods."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 0, "scale": 0.1},
            seasonality=[7, 14],
            seasonality_strength=2.0,  # Single value for multiple periods
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 100

    def test_simulator_different_frequencies(self):
        """Test different time frequencies."""
        for freq in ["D", "W", "h"]:
            sim = TimeSeriesSimulator(length=50, freq=freq, random_state=42)
            y = sim.simulate()

            assert len(y) == 50
            assert isinstance(y.index, pd.DatetimeIndex)

    def test_simulator_exponential_distribution(self):
        """Test exponential distribution."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="exponential",
            dist_params={"scale": 2.0},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert (y >= 0).all()

    def test_simulator_gamma_distribution(self):
        """Test gamma distribution."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="gamma",
            dist_params={"shape": 2.0, "scale": 2.0},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert (y >= 0).all()

    def test_simulator_uniform_distribution(self):
        """Test uniform distribution."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="uniform",
            dist_params={"low": 0, "high": 10},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert (y >= 0).all() and (y <= 10).all()

    def test_simulator_binomial_distribution(self):
        """Test binomial distribution."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="binomial",
            dist_params={"n": 10, "p": 0.5},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert (y >= 0).all() and (y <= 10).all()

    def test_simulator_lognormal_distribution(self):
        """Test lognormal distribution."""
        sim = TimeSeriesSimulator(
            length=50,
            distribution="lognormal",
            dist_params={"mean": 0, "sigma": 1},
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert (y > 0).all()

    def test_simulator_invalid_distribution(self):
        """Test invalid distribution name."""
        sim = TimeSeriesSimulator(distribution="invalid_dist", random_state=42)

        with pytest.raises(ValueError, match="Unknown distribution"):
            sim.simulate()

    def test_simulator_invalid_trend(self):
        """Test invalid trend name."""
        sim = TimeSeriesSimulator(trend="invalid_trend", random_state=42)

        with pytest.raises(ValueError, match="Unknown trend"):
            sim.simulate()

    def test_simulator_invalid_length(self):
        """Test invalid length parameter."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            TimeSeriesSimulator(length=0)

        with pytest.raises(ValueError, match="length must be >= 1"):
            TimeSeriesSimulator(length=-5)

    def test_simulator_seasonality_strength_mismatch(self):
        """Test mismatched seasonality and strength lengths."""
        sim = TimeSeriesSimulator(
            length=100,
            seasonality=[7, 14, 30],
            seasonality_strength=[1.0, 2.0],  # Wrong length
            random_state=42,
        )

        with pytest.raises(ValueError, match="seasonality_strength must be"):
            sim.simulate()

    def test_simulator_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        sim1 = TimeSeriesSimulator(length=50, random_state=42)
        y1 = sim1.simulate()

        sim2 = TimeSeriesSimulator(length=50, random_state=42)
        y2 = sim2.simulate()

        pd.testing.assert_series_equal(y1, y2)

    def test_simulator_different_random_states(self):
        """Test that different random states produce different results."""
        sim1 = TimeSeriesSimulator(length=50, random_state=42)
        y1 = sim1.simulate()

        sim2 = TimeSeriesSimulator(length=50, random_state=123)
        y2 = sim2.simulate()

        # Values should be different
        assert not np.allclose(y1.values, y2.values)

    def test_simulator_complex_scenario(self):
        """Test complex scenario with all components."""
        sim = TimeSeriesSimulator(
            length=200,
            distribution="poisson",
            dist_params={"lam": 10},
            trend="linear",
            trend_params={"slope": 0.05},
            seasonality=[7, 30],
            seasonality_strength=[2.0, 1.5],
            noise_std=1.0,
            freq="D",
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 200
        assert isinstance(y, pd.Series)
        assert isinstance(y.index, pd.DatetimeIndex)

    def test_simulator_custom_start_date(self):
        """Test custom start date."""
        sim = TimeSeriesSimulator(
            length=50,
            start="2023-06-15",
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 50
        assert y.index[0] == pd.Timestamp("2023-06-15")

    def test_simulator_auto_length_adjustment(self):
        """Test automatic length adjustment for seasonality."""
        # Request length=20 with seasonality=12
        # Should auto-adjust to 36 (3 * 12)
        sim = TimeSeriesSimulator(
            length=20,
            seasonality=12,
            random_state=42,
        )
        y = sim.simulate()

        assert len(y) == 36  # 3 * 12

    def test_simulator_time_index_attribute(self):
        """Test that time_index_ attribute is set after simulate."""
        sim = TimeSeriesSimulator(length=50, random_state=42)
        y = sim.simulate()

        assert hasattr(sim, "time_index_")
        assert len(sim.time_index_) == 50
        pd.testing.assert_index_equal(y.index, sim.time_index_)

    def test_get_test_params(self):
        """Test get_test_params class method."""
        params_list = TimeSeriesSimulator.get_test_params()

        assert isinstance(params_list, list)
        assert len(params_list) >= 1

        # Each params should create a valid simulator
        for params in params_list:
            sim = TimeSeriesSimulator(**params)
            y = sim.simulate()
            assert isinstance(y, pd.Series)
