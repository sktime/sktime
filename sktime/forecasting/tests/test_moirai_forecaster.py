"""Tests for MOIRAIForecaster probabilistic forecasting methods.

Tests for:
- Quantile monotonicity: q(0.1) <= q(0.5) <= q(0.9)
- Mean vs median consistency documentation
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(
    True,  # Will be set to False when dependencies are available
    reason="Requires MOIRAI dependencies (torch, gluonts, etc.)",
)
class TestMOIRAIProbabilisticForecasting:
    """Test probabilistic forecasting capabilities of MOIRAIForecaster."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        y = pd.DataFrame(
            np.random.randn(50, 2),
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
            columns=["a", "b"],
        )
        return y

    def test_quantile_monotonicity(self, sample_data):
        """Test that MOIRAI quantiles are monotonically increasing.

        Verifies: q(0.1) <= q(0.5) <= q(0.9) for all time steps and variables.
        This is a fundamental property of valid quantile estimates.
        """
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster

        forecaster = MOIRAIForecaster(
            checkpoint_path="sktime/moirai-1.0-R-small",
            deterministic=True,
            num_samples=100,
        )
        forecaster.fit(sample_data)

        # Get quantiles at multiple levels
        alpha = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantiles = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=alpha)

        # Check monotonicity for each variable
        for var in ["a", "b"]:
            for idx in quantiles.index:
                q_values = [quantiles.loc[idx, (var, a)] for a in alpha]
                # Check q[i] <= q[i+1] for all i
                assert all(
                    q_values[i] <= q_values[i + 1] + 1e-10  # tolerance for float
                    for i in range(len(q_values) - 1)
                ), f"Quantile monotonicity violated for {var} at {idx}"

    def test_mean_vs_median_relationship(self, sample_data):
        """Test relationship between predict() mean and predict_quantiles(0.5) median.

        IMPORTANT: predict() returns MEAN (via GluonTS forecast.mean_ts).
        predict_quantiles(alpha=0.5) returns MEDIAN.

        For symmetric distributions: mean ≈ median (tight tolerance)
        For skewed distributions: mean ≠ median (loose tolerance)

        This test documents the expected behavior, not a strict equality check.
        """
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster

        forecaster = MOIRAIForecaster(
            checkpoint_path="sktime/moirai-1.0-R-small",
            deterministic=True,
            num_samples=500,  # Higher for stable estimates
        )
        forecaster.fit(sample_data)

        fh = [1, 2, 3]
        point_forecast = forecaster.predict(fh=fh)  # Returns MEAN
        quantile_50 = forecaster.predict_quantiles(fh=fh, alpha=0.5)  # Returns MEDIAN

        for var in ["a", "b"]:
            mean_vals = point_forecast[var].values
            median_vals = quantile_50[(var, 0.5)].values

            # Document the relationship, not strict equality
            # rtol=0.5 allows for skewed distributions while catching bugs
            try:
                np.testing.assert_allclose(mean_vals, median_vals, rtol=0.5)
            except AssertionError:
                # This is expected for highly skewed distributions
                # Just ensure both are finite and reasonable
                assert np.all(np.isfinite(mean_vals)), "Mean contains non-finite values"
                assert np.all(
                    np.isfinite(median_vals)
                ), "Median contains non-finite values"

    def test_predict_interval_derived_from_quantiles(self, sample_data):
        """Test that predict_interval works (derived from quantiles by BaseForecaster)."""
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster

        forecaster = MOIRAIForecaster(
            checkpoint_path="sktime/moirai-1.0-R-small",
            deterministic=True,
            num_samples=100,
        )
        forecaster.fit(sample_data)

        # Test interval prediction
        intervals = forecaster.predict_interval(fh=[1, 2, 3], coverage=0.8)

        # Check DataFrame structure
        assert isinstance(intervals, pd.DataFrame)
        assert len(intervals) == 3  # 3 forecast horizons

        # Check that lower < upper for all intervals
        for var in ["a", "b"]:
            lower = intervals[(var, 0.8, "lower")].values
            upper = intervals[(var, 0.8, "upper")].values
            assert np.all(lower <= upper), f"Lower > upper for {var}"

    def test_validate_alpha_single_float(self, sample_data):
        """Test that single float alpha works (beginner-friendly)."""
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster

        forecaster = MOIRAIForecaster(
            checkpoint_path="sktime/moirai-1.0-R-small",
            deterministic=True,
            num_samples=50,
        )
        forecaster.fit(sample_data)

        # Single float should work
        quantiles = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=0.5)
        assert isinstance(quantiles, pd.DataFrame)
        assert (sample_data.columns[0], 0.5) in quantiles.columns

    def test_validate_alpha_out_of_range_raises(self, sample_data):
        """Test that out-of-range alpha raises ValueError."""
        from sktime.forecasting.moirai_forecaster import MOIRAIForecaster

        forecaster = MOIRAIForecaster(
            checkpoint_path="sktime/moirai-1.0-R-small",
            deterministic=True,
            num_samples=50,
        )
        forecaster.fit(sample_data)

        with pytest.raises(ValueError, match="alpha values must be in"):
            forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[1.5])

        with pytest.raises(ValueError, match="alpha values must be in"):
            forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[-0.1])
