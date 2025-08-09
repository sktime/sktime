"""Tests result saving capabilities of benchmarking module."""

__author__ = ["jgyasu"]

from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.split import SingleWindowSplitter


def test_benchmark_output(tmp_path):
    """Test uniqueness of model-task pair across different benchmark runs."""
    output_file = tmp_path / "results.csv"

    splitter = SingleWindowSplitter(window_length=16, fh=[1, 2, 3])
    scorers = [MeanSquaredError(square_root=True)]
    dataset_loader = load_airline()

    benchmark1 = ForecastingBenchmark()

    benchmark1.add_task(dataset_loader, splitter, scorers)
    benchmark1.add_estimator(
        NaiveForecaster(strategy="mean", sp=4), estimator_id="MEAN"
    )

    results1 = benchmark1.run(output_file=output_file)

    assert len(results1) == 1

    benchmark2 = ForecastingBenchmark()

    benchmark2.add_task(dataset_loader, splitter, scorers)
    benchmark2.add_estimator(
        NaiveForecaster(strategy="mean", sp=4), estimator_id="MEAN"
    )

    results2 = benchmark2.run(output_file=output_file)

    assert len(results2) == 1
