# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ForecastingBenchmark."""

__author__ = ["sktime developers"]

import numpy as np
import pandas as pd
import pytest

from sktime.benchmarking.forecasting import ForecastingBenchmark, TimeSeriesSimulator
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_for_class(ForecastingBenchmark),
    reason="Run tests for ForecastingBenchmark only if explicitly requested",
)
class TestForecastingBenchmark:
    """Tests for ForecastingBenchmark class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        sim = TimeSeriesSimulator(
            length=100,
            distribution="normal",
            dist_params={"loc": 10, "scale": 2},
            random_state=42,
        )
        return sim.simulate()

    def test_benchmark_with_specified_models(self, sample_data):
        """Test benchmark with user-specified models."""
        models = [
            ("naive_last", NaiveForecaster(strategy="last")),
            ("naive_mean", NaiveForecaster(strategy="mean")),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False, random_state=42
        )

        results = benchmark.run(sample_data)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert "mae" in results.columns
        assert "mse" in results.columns
        assert "mape" in results.columns
        assert "naive_last" in results.index
        assert "naive_mean" in results.index

    def test_benchmark_with_tuple_models(self, sample_data):
        """Test that models can be provided as tuples."""
        models = [
            ("naive", NaiveForecaster()),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        results = benchmark.run(sample_data)
        assert len(results) == 1

    def test_benchmark_with_instance_models(self, sample_data):
        """Test that models can be provided as instances."""
        models = [NaiveForecaster()]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        results = benchmark.run(sample_data)
        assert len(results) == 1

    def test_benchmark_custom_metrics(self, sample_data):
        """Test benchmark with custom metrics."""

        def custom_metric(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        models = [("naive", NaiveForecaster())]
        benchmark = ForecastingBenchmark(
            models=models,
            fh=1,
            metrics=[custom_metric],
            test_size=20,
            verbose=False,
        )

        results = benchmark.run(sample_data)
        assert "custom_metric" in results.columns

    def test_benchmark_multiple_horizons(self, sample_data):
        """Test benchmark with multiple forecast horizons."""
        models = [("naive", NaiveForecaster())]

        benchmark = ForecastingBenchmark(
            models=models, fh=[1, 2, 3], test_size=20, verbose=False
        )

        results = benchmark.run(sample_data)
        assert len(results) == 1

    def test_benchmark_test_size_int(self, sample_data):
        """Test benchmark with integer test size."""
        models = [("naive", NaiveForecaster())]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=15, verbose=False
        )

        results = benchmark.run(sample_data)
        assert not results.empty

    def test_benchmark_test_size_float(self, sample_data):
        """Test benchmark with float test size."""
        models = [("naive", NaiveForecaster())]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=0.3, verbose=False
        )

        results = benchmark.run(sample_data)
        assert not results.empty

    def test_benchmark_get_best_model(self, sample_data):
        """Test getting the best model."""
        models = [
            ("naive_last", NaiveForecaster(strategy="last")),
            ("naive_mean", NaiveForecaster(strategy="mean")),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        benchmark.run(sample_data)
        best_name, best_model, best_score = benchmark.get_best_model(metric="mae")

        assert isinstance(best_name, str)
        assert isinstance(best_model, NaiveForecaster)
        assert isinstance(best_score, (int, float, np.number))
        assert not np.isnan(best_score)

    def test_benchmark_get_best_model_before_run(self):
        """Test that get_best_model fails before running benchmark."""
        benchmark = ForecastingBenchmark(verbose=False)

        with pytest.raises(ValueError, match="Must run benchmark"):
            benchmark.get_best_model()

    def test_benchmark_stores_fitted_models(self, sample_data):
        """Test that fitted models are stored."""
        models = [("naive", NaiveForecaster())]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        benchmark.run(sample_data)

        assert "naive" in benchmark.fitted_models_
        assert isinstance(benchmark.fitted_models_["naive"], NaiveForecaster)

    def test_benchmark_stores_predictions(self, sample_data):
        """Test that predictions are stored."""
        models = [("naive", NaiveForecaster())]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        benchmark.run(sample_data)

        assert "naive" in benchmark.predictions_
        assert isinstance(benchmark.predictions_["naive"], pd.Series)

    def test_benchmark_handles_model_errors(self):
        """Test that benchmark handles model failures gracefully."""

        class FailingForecaster(NaiveForecaster):
            def _fit(self, y, X=None, fh=None):
                raise RuntimeError("Intentional failure")

        # Create simple data
        y = pd.Series(
            np.random.randn(50), index=pd.date_range("2020-01-01", periods=50)
        )

        models = [
            ("failing", FailingForecaster()),
            ("naive", NaiveForecaster()),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=10, verbose=False
        )

        results = benchmark.run(y)

        # Failing model should have NaN results
        assert pd.isna(results.loc["failing", "mae"])
        # Naive model should work
        assert not pd.isna(results.loc["naive", "mae"])
        # Error should be stored
        assert "failing" in benchmark.errors_

    def test_benchmark_with_theta(self, sample_data):
        """Test benchmark with Theta forecaster."""
        if not _check_soft_dependencies("statsmodels", severity="none"):
            pytest.skip("statsmodels not available")

        models = [
            ("naive", NaiveForecaster()),
            ("theta", ThetaForecaster()),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False
        )

        results = benchmark.run(sample_data)

        assert len(results) == 2
        assert "theta" in results.index

    def test_benchmark_invalid_model_type(self, sample_data):
        """Test that invalid model types raise errors."""
        models = ["not_a_model"]

        benchmark = ForecastingBenchmark(models=models, verbose=False)

        with pytest.raises(ValueError, match="must be BaseForecaster"):
            benchmark.run(sample_data)

    def test_benchmark_empty_models_list(self):
        """Test with empty models list."""
        y = pd.Series(
            np.random.randn(50), index=pd.date_range("2020-01-01", periods=50)
        )

        benchmark = ForecastingBenchmark(models=[], verbose=False)

        with pytest.raises(ValueError, match="No models available"):
            benchmark.run(y)

    def test_benchmark_reproducibility(self, sample_data):
        """Test that random_state ensures reproducible results."""
        models = [("naive", NaiveForecaster())]

        benchmark1 = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False, random_state=42
        )
        results1 = benchmark1.run(sample_data)

        benchmark2 = ForecastingBenchmark(
            models=models, fh=1, test_size=20, verbose=False, random_state=42
        )
        results2 = benchmark2.run(sample_data)

        pd.testing.assert_frame_equal(results1, results2)

    def test_benchmark_with_simulated_poisson_data(self):
        """Test complete workflow with simulated Poisson data."""
        # Generate Poisson-distributed time series
        sim = TimeSeriesSimulator(
            length=150,
            distribution="poisson",
            dist_params={"lam": 10},
            trend="linear",
            trend_params={"slope": 0.05},
            random_state=42,
        )
        y = sim.simulate()

        # Benchmark models
        models = [
            ("naive_last", NaiveForecaster(strategy="last")),
            ("naive_mean", NaiveForecaster(strategy="mean")),
        ]

        benchmark = ForecastingBenchmark(
            models=models, fh=[1, 2, 3], test_size=30, verbose=False
        )

        results = benchmark.run(y)

        assert len(results) == 2
        assert all(col in results.columns for col in ["mae", "mse", "mape"])

        # Get best model
        best_name, best_model, best_score = benchmark.get_best_model()
        assert best_name in ["naive_last", "naive_mean"]
