"""Tests for Detection Benchmark."""

__author__ = ["Nischal1425"]

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from sktime.benchmarking.detection import DetectionBenchmark
from sktime.detection.dummy import (
    DummyRegularAnomalies,
    DummyRegularChangePoints,
)
from sktime.performance_metrics.detection import (
    DirectedChamfer,
    WindowedF1Score,
)
from sktime.tests.test_switch import run_test_module_changed


def _make_detection_data():
    """Create synthetic detection data for benchmarking tests."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"value": rng.randn(100)})
    y = pd.DataFrame({"ilocs": list(range(9, 100, 10))})
    return X, y


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("write_file", [True, False])
def test_detection_benchmark(tmp_path, write_file):
    """Test detection benchmark with single estimator and task."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(_make_detection_data, cv_splitter, scorers)

    if write_file:
        results_file = tmp_path / "results.csv"
    else:
        results_file = None
    results_df = benchmark.run(results_file)

    expected_benchmark_labels = [
        "WindowedF1Score_fold_0_test",
        "WindowedF1Score_fold_1_test",
        "WindowedF1Score_fold_2_test",
        "WindowedF1Score_mean",
        "WindowedF1Score_std",
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
def test_add_list_estimators():
    """Test adding list of estimators."""
    benchmark = DetectionBenchmark()

    estimators = [
        DummyRegularAnomalies(step_size=5),
        DummyRegularChangePoints(step_size=3),
    ]

    benchmark.add_estimator(estimators)
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(_make_detection_data, cv_splitter, scorers)
    results_df = benchmark.run()

    # should have 2 rows (one per estimator)
    assert len(results_df) == 2


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_dict_estimators():
    """Test adding dict of estimators with custom IDs."""
    benchmark = DetectionBenchmark()

    estimators = {
        "anomaly_5": DummyRegularAnomalies(step_size=5),
        "cp_3": DummyRegularChangePoints(step_size=3),
    }

    benchmark.add_estimator(estimators)
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(_make_detection_data, cv_splitter, scorers)
    results_df = benchmark.run()

    assert len(results_df) == 2
    model_ids = results_df["model_id"].tolist()
    assert "anomaly_5" in model_ids
    assert "cp_3" in model_ids


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_multiple_metrics():
    """Test detection benchmark with multiple metrics."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2), DirectedChamfer()]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(_make_detection_data, cv_splitter, scorers)
    results_df = benchmark.run()

    result_rows = results_df.T.index.to_list()
    assert "WindowedF1Score_mean" in result_rows
    assert "DirectedChamfer_mean" in result_rows


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_multiple_tasks():
    """Test detection benchmark with multiple tasks."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2)]

    cv_kfold = KFold(n_splits=3, shuffle=False)
    cv_ts = TimeSeriesSplit(n_splits=3)

    benchmark.add_task(_make_detection_data, cv_kfold, scorers, task_id="kfold_task")
    benchmark.add_task(_make_detection_data, cv_ts, scorers, task_id="ts_task")
    results_df = benchmark.run()

    # should have 2 rows (one per task)
    assert len(results_df) == 2
    validation_ids = results_df["validation_id"].tolist()
    assert "kfold_task" in validation_ids
    assert "ts_task" in validation_ids


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_unsupervised_detection_benchmark():
    """Test detection benchmark with unsupervised detector (no y)."""

    def make_unsupervised_data():
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"value": rng.randn(100)})
        return X, None

    benchmark = DetectionBenchmark()
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(make_unsupervised_data, cv_splitter, scorers)
    results_df = benchmark.run()

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 1


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_custom_task_id():
    """Test detection benchmark with custom task ID."""
    benchmark = DetectionBenchmark()
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(
        _make_detection_data, cv_splitter, scorers, task_id="my_custom_task"
    )
    results_df = benchmark.run()

    assert results_df["validation_id"].iloc[0] == "my_custom_task"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_return_data():
    """Test detection benchmark with return_data=True."""
    benchmark = DetectionBenchmark(return_data=True)
    benchmark.add_estimator(DummyRegularAnomalies(step_size=5))
    scorers = [WindowedF1Score(margin=2)]

    cv_splitter = KFold(n_splits=3, shuffle=False)
    benchmark.add_task(_make_detection_data, cv_splitter, scorers)
    results_df = benchmark.run()

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 1
