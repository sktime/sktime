# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TimeSeriesSimulator."""

__author__ = ["sktime developers"]

import pandas as pd
import pytest

from sktime.benchmarking.forecasting import TimeSeriesSimulator
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesSimulator),
    reason="Run tests for TimeSeriesSimulator only if explicitly requested",
)
class TestTimeSeriesSimulator:
    """Tests for TimeSeriesSimulator class."""

    def test_simulator_default(self):
        """Test simulator with default parameters."""
        sim = TimeSeriesSimulator()
        y = sim.simulate()

        assert isinstance(y, pd.Series)
        assert len(y) == 100
        assert isinstance(y.index, pd.DatetimeIndex)

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
        # Seasonal pattern should show some periodicity

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

        # With noise
        sim2 = TimeSeriesSimulator(
            length=50,
            distribution="normal",
            dist_params={"loc": 10, "scale": 1},
            noise_std=5.0,
            random_state=42,
        )
        y2 = sim2.simulate()

        # Series with more noise should have higher variance
        assert y2.var() > y1.var()

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

    def test_simulator_different_frequencies(self):
        """Test different time frequencies."""
        for freq in ["D", "W", "M", "H"]:
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

    def test_simulator_invalid_distribution(self):
        """Test invalid distribution name."""
        sim = TimeSeriesSimulator(distribution="invalid_dist")

        with pytest.raises(ValueError, match="Unknown distribution"):
            sim.simulate()

    def test_simulator_invalid_trend(self):
        """Test invalid trend name."""
        sim = TimeSeriesSimulator(trend="invalid_trend")

        with pytest.raises(ValueError, match="Unknown trend"):
            sim.simulate()

    def test_simulator_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        sim1 = TimeSeriesSimulator(length=50, random_state=42)
        y1 = sim1.simulate()

        sim2 = TimeSeriesSimulator(length=50, random_state=42)
        y2 = sim2.simulate()

        pd.testing.assert_series_equal(y1, y2)

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
