"""Tests for Regression Benchmark."""

__author__ = ["NAME-ASHWANIYADAV"]

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from sktime.benchmarking.regression import RegressionBenchmark
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.dummy import DummyRegressor
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.panel import make_regression_problem


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("write_file", [True, False])
def test_regression_benchmark(tmp_path, write_file):
    """Test regression benchmark with single estimator and task."""
    benchmark = RegressionBenchmark()
    benchmark.add_estimator(DummyRegressor())
    scorers = [mean_squared_error]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_regression_problem, cv_splitter, scorers)

    if write_file:
        results_file = tmp_path / "results.csv"
    else:
        results_file = None
    results_df = benchmark.run(results_file)

    expected_benchmark_labels = [
        "mean_squared_error_fold_0_test",
        "mean_squared_error_fold_1_test",
        "mean_squared_error_fold_2_test",
        "mean_squared_error_mean",
        "mean_squared_error_std",
        "fit_time_fold_0_test",
        "fit_time_fold_1_test",
        "fit_time_fold_2_test",
        "fit_time_mean",
        "fit_time_std",
        "pred_time_fold_0_test",
        "pred_time_fold_1_test",
        "pred_time_fold_2_test",
        "pred_time_mean",
        "pred_time_std",
        "runtime_secs",
    ]

    result_rows = results_df.T.index.to_list()
    for metric in expected_benchmark_labels:
        assert metric in result_rows, f"{metric} not found in result rows"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_list_estimators(tmp_path):
    """Test adding list of estimators."""
    benchmark = RegressionBenchmark()

    estimators = [DummyRegressor(), KNeighborsTimeSeriesRegressor()]

    benchmark.add_estimator(estimators)
    scorers = [mean_squared_error, mean_absolute_error]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_regression_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["DummyRegressor", "KNeighborsTimeSeriesRegressor"], name="model_id"),
        results_df["model_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_dict_estimators(tmp_path):
    """Test adding dict of estimators."""
    benchmark = RegressionBenchmark()

    estimators = {"D": DummyRegressor(), "KN": KNeighborsTimeSeriesRegressor()}

    benchmark.add_estimator(estimators)
    scorers = [mean_squared_error, mean_absolute_error]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_regression_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["D", "KN"], name="model_id"),
        results_df["model_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_estimator_twice(tmp_path):
    """Test adding the same estimator twice."""
    benchmark = RegressionBenchmark()
    benchmark.add_estimator(DummyRegressor())
    benchmark.add_estimator(DummyRegressor())
    scorers = [mean_squared_error]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_regression_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["DummyRegressor", "DummyRegressor_2"], name="model_id"),
        results_df["model_id"],
    )

    msg = "add_estimator does not register all estimators."
    assert len(benchmark.estimators.entities) == 2, msg


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_multiple_tasks(tmp_path):
    """Test adding multiple tasks for benchmarking."""
    benchmark = RegressionBenchmark()
    benchmark.add_estimator(DummyRegressor())

    dataset_loaders = [make_regression_problem]
    cv_splitter = KFold(n_splits=3)
    scorers = [mean_squared_error, mean_absolute_error]

    for dataset_loader in dataset_loaders:
        benchmark.add_task(
            dataset_loader,
            cv_splitter,
            scorers,
        )

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(
            [
                "[dataset=make_regression_problem]_[cv_splitter=KFold]",
            ],
            name="validation_id",
        ),
        results_df["validation_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_multiple_metrics(tmp_path):
    """Test benchmark with multiple metrics produces all expected columns."""
    benchmark = RegressionBenchmark()
    benchmark.add_estimator(DummyRegressor())
    scorers = [mean_squared_error, mean_absolute_error]

    cv_splitter = KFold(n_splits=2)
    benchmark.add_task(make_regression_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    result_rows = results_df.T.index.to_list()

    # Check both metrics are present
    assert "mean_squared_error_mean" in result_rows
    assert "mean_absolute_error_mean" in result_rows
