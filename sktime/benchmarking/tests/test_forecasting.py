# -*- coding: utf-8 -*-
"""Forecasting benchmarks tests."""

import pandas as pd
import pytest

from sktime.benchmarking import forecasting
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError
from sktime.utils.validation._dependencies import _check_soft_dependencies


def data_loader_simple() -> pd.DataFrame:
    """Return simple data for use in testing."""
    return pd.DataFrame([2, 2, 3])


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_forecastingbenchmark(tmp_path):
    """Test benchmarking a forecaster estimator."""
    benchmark = forecasting.ForecastingBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="last"))

    cv_splitter = ExpandingWindowSplitter(
        initial_window=1,
        step_length=1,
        fh=1,
    )
    benchmark.add_task(data_loader_simple, cv_splitter, [MeanSquaredPercentageError()])

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    results_df = results_df.drop(columns=["runtime_secs"])

    expected_results_df = pd.DataFrame(
        [
            (
                (
                    "[dataset=data_loader_simple]_"
                    "[cv_splitter=ExpandingWindowSplitter]-v1"
                ),
                "NaiveForecaster-v1",
                0.0,
                0.111,
                0.0555,
                0.0785,
            )
        ],
        columns=[
            "validation_id",
            "model_id",
            "MeanSquaredPercentageError_fold_0_test",
            "MeanSquaredPercentageError_fold_1_test",
            "MeanSquaredPercentageError_mean",
            "MeanSquaredPercentageError_std",
        ],
    )

    pd.testing.assert_frame_equal(
        expected_results_df, results_df, check_exact=False, atol=0, rtol=0.001
    )
