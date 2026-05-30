"""Test checkpointing functionality in benchmark experiments."""

import pandas as pd
import pytest

from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.datasets import Airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError
from sktime.split import SingleWindowSplitter


def _setup_benchmark():
    """Create an experiment with two estimators."""
    benchmark = ForecastingBenchmark()

    benchmark.add_estimator(
        NaiveForecaster(strategy="last"),
        estimator_id="naive_last",
    )

    benchmark.add_estimator(
        NaiveForecaster(strategy="mean"),
        estimator_id="naive_mean",
    )

    benchmark.add_task(
        Airline(),
        SingleWindowSplitter(window_length=52, fh=range(1, 5)),
        [MeanSquaredPercentageError()],
    )

    return benchmark


def test_checkpoint_saved_when_benchmark_crashes(tmp_path, monkeypatch):
    """Completed experiments should be saved before a later failure."""
    benchmark = _setup_benchmark()

    original_run_validation = benchmark._run_validation
    n_calls = 0

    def crashing_run_validation(task, estimator):
        nonlocal n_calls

        n_calls += 1

        if n_calls == 2:
            raise RuntimeError("Simulated benchmark failure")

        return original_run_validation(task, estimator)

    monkeypatch.setattr(
        benchmark,
        "_run_validation",
        crashing_run_validation,
    )

    results_path = tmp_path / "results.csv"

    with pytest.raises(RuntimeError, match="Simulated benchmark failure"):
        benchmark.run(str(results_path))

    assert results_path.exists()

    results = pd.read_csv(results_path)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1
