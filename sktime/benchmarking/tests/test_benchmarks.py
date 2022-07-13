# -*- coding: utf-8 -*-
"""Benchmarks tests."""
from typing import Callable

import pandas as pd

from sktime.base import BaseEstimator
from sktime.benchmarking import benchmarks
from sktime.forecasting.naive import NaiveForecaster


def factory_estimator_class_task() -> Callable:
    """Return task for getting class of estimator."""

    def estimator_class_task(estimator: BaseEstimator) -> str:
        """Return class of estimator.

        Used as simple task for testing purposes.
        """
        return {"estimator_class": str(estimator.__class__)}

    return estimator_class_task


def test_basebenchmark(tmp_path):
    """Test registering estimator, registering a simple task, and running."""
    benchmark = benchmarks.BaseBenchmark()

    benchmark.add_estimator(NaiveForecaster)
    benchmark._add_task(factory_estimator_class_task)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    results_df = results_df.drop(columns=["runtime_secs"])

    expected_results_df = pd.DataFrame(
        [
            (
                "factory_estimator_class_task-v1",
                "NaiveForecaster-v1",
                "<class 'sktime.forecasting.naive.NaiveForecaster'>",
            )
        ],
        columns=["validation_id", "model_id", "estimator_class"],
    )

    pd.testing.assert_frame_equal(expected_results_df, results_df)
    pd.testing.assert_frame_equal(
        expected_results_df, pd.read_csv(results_file).drop(columns=["runtime_secs"])
    )
