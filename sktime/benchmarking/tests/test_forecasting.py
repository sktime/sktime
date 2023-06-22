# -*- coding: utf-8 -*-
"""Forecasting benchmarks tests."""

import pandas as pd
import pytest

from sktime.benchmarking import forecasting
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredPercentageError,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

expected_results_df_1 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_simple]_"
        + "[cv_splitter=ExpandingWindowSplitter]-v1",
        "model_id": "NaiveForecaster-v1",
        "MeanSquaredPercentageError_fold_0_test": 0.0,
        "MeanSquaredPercentageError_fold_1_test": 0.111,
        "MeanSquaredPercentageError_mean": 0.0555,
        "MeanSquaredPercentageError_std": 0.0785,
    },
    index=[0],
)

expected_results_df_2 = pd.DataFrame(
    data={
        "validation_id": "[dataset=data_loader_simple]_"
        + "[cv_splitter=ExpandingWindowSplitter]-v1",
        "model_id": "NaiveForecaster-v1",
        "MeanAbsolutePercentageError_fold_0_test": 0.0,
        "MeanAbsolutePercentageError_fold_1_test": 0.333,
        "MeanAbsolutePercentageError_mean": 0.1666,
        "MeanAbsolutePercentageError_std": 0.2357,
        "MeanAbsoluteError_fold_0_test": 0.0,
        "MeanAbsoluteError_fold_1_test": 1.0,
        "MeanAbsoluteError_mean": 0.5,
        "MeanAbsoluteError_std": 0.7071,
    },
    index=[0],
)


def data_loader_simple() -> pd.DataFrame:
    """Return simple data for use in testing."""
    return pd.DataFrame([2, 2, 3])


@pytest.mark.skipif(
    not _check_soft_dependencies("kotsu", severity="none"),
    reason="skip test if required soft dependencies not available",
)
@pytest.mark.parametrize(
    "expected_results_df, scorers",
    [
        (expected_results_df_1, [MeanSquaredPercentageError()]),
        (expected_results_df_2, [MeanAbsolutePercentageError(), MeanAbsoluteError()]),
    ],
)
def test_forecastingbenchmark(tmp_path, expected_results_df, scorers):
    """Test benchmarking a forecaster estimator."""
    benchmark = forecasting.ForecastingBenchmark()

    benchmark.add_estimator(NaiveForecaster(strategy="last"))

    cv_splitter = ExpandingWindowSplitter(
        initial_window=1,
        step_length=1,
        fh=1,
    )
    benchmark.add_task(data_loader_simple, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    results_df = results_df.drop(columns=["runtime_secs"])

    pd.testing.assert_frame_equal(
        expected_results_df, results_df, check_exact=False, atol=0, rtol=0.001
    )
