"""Tests for fault-tolerant benchmark execution."""

import logging

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.benchmarking.benchmarks import FailedExperimentRecord
from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter
from sktime.utils._testing.panel import make_classification_problem


class _FailingClassifier(DummyClassifier):
    """Classifier that always raises on fit."""

    def _fit(self, X, y):
        raise RuntimeError("classifier fit failed")


class _FailingForecaster(NaiveForecaster):
    """Forecaster that always raises on fit."""

    def _fit(self, y, X, fh):
        raise ValueError("forecaster fit failed")


def _data_loader_simple() -> pd.DataFrame:
    return pd.DataFrame([2, 2, 3])


@pytest.fixture
def cv_splitter_forecasting():
    return ExpandingWindowSplitter(initial_window=1, step_length=1, fh=1)


@pytest.fixture
def cv_splitter_classification():
    return KFold(n_splits=3)


def test_classification_continues_after_failure(
    tmp_path, cv_splitter_classification, caplog
):
    """Failed classification pair is recorded with NaN; remaining pairs still run."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )

    with caplog.at_level(logging.WARNING):
        results_df = benchmark.run(tmp_path / "results.csv")

    assert len(results_df) == 2
    assert set(results_df["model_id"]) == {"DummyClassifier", "_FailingClassifier"}

    failing_row = results_df.loc[results_df["model_id"] == "_FailingClassifier"].iloc[0]
    assert pd.isna(failing_row["accuracy_score_mean"])
    assert pd.isna(failing_row["fit_time_mean"])

    success_row = results_df.loc[results_df["model_id"] == "DummyClassifier"].iloc[0]
    assert not pd.isna(success_row["accuracy_score_mean"])

    failures = benchmark.failed_experiments
    assert len(failures) == 1
    assert failures[0] == FailedExperimentRecord(
        task_id="[dataset=make_classification_problem]_[cv_splitter=KFold]",
        model_id="_FailingClassifier",
        exception_type="RuntimeError",
        exception_message="classifier fit failed",
    )

    assert "Benchmark completed with 1 failed task-estimator pair(s)" in caplog.text
    assert "RuntimeError: classifier fit failed" in caplog.text


def test_forecasting_continues_after_failure(tmp_path, cv_splitter_forecasting, caplog):
    """Failed forecasting pair is recorded with NaN; remaining pairs still run."""
    benchmark = ForecastingBenchmark()
    benchmark.add_estimator(_FailingForecaster())
    benchmark.add_estimator(NaiveForecaster(strategy="last"))
    benchmark.add_task(
        _data_loader_simple, cv_splitter_forecasting, [MeanAbsoluteError()]
    )

    with caplog.at_level(logging.WARNING):
        results_df = benchmark.run(tmp_path / "results.csv")

    assert len(results_df) == 2
    assert set(results_df["model_id"]) == {"NaiveForecaster", "_FailingForecaster"}

    failing_row = results_df.loc[results_df["model_id"] == "_FailingForecaster"].iloc[0]
    assert pd.isna(failing_row["MeanAbsoluteError_mean"])
    assert pd.isna(failing_row["fit_time_mean"])

    failures = benchmark.failed_experiments
    assert len(failures) == 1
    assert failures[0].exception_type == "ValueError"
    assert failures[0].exception_message == "forecaster fit failed"

    assert "Benchmark completed with 1 failed task-estimator pair(s)" in caplog.text


def test_multiple_failures_all_reported(tmp_path, cv_splitter_classification, caplog):
    """Multiple failing pairs are all recorded and reported."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )
    benchmark.add_task(
        make_classification_problem,
        KFold(n_splits=2),
        [accuracy_score],
        task_id="task-2",
    )

    with caplog.at_level(logging.WARNING):
        results_df = benchmark.run(tmp_path / "results.csv")

    assert len(results_df) == 4
    assert set(results_df["model_id"]) == {"DummyClassifier", "_FailingClassifier"}

    failing_rows = results_df.loc[results_df["model_id"] == "_FailingClassifier"]
    assert len(failing_rows) == 2
    assert failing_rows["accuracy_score_mean"].isna().all()

    assert len(benchmark.failed_experiments) == 2
    assert {f.model_id for f in benchmark.failed_experiments} == {"_FailingClassifier"}
    assert {f.task_id for f in benchmark.failed_experiments} == {
        "[dataset=make_classification_problem]_[cv_splitter=KFold]",
        "task-2",
    }
    assert "Benchmark completed with 2 failed task-estimator pair(s)" in caplog.text


def test_failed_pair_saved_to_checkpoint_with_nan(tmp_path, cv_splitter_classification):
    """Failed pairs are persisted with NaN scores; successful pairs are persisted."""
    results_file = tmp_path / "results.csv"

    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )
    benchmark.run(results_file)

    saved_df = pd.read_csv(results_file)
    assert len(saved_df) == 2
    assert set(saved_df["model_id"]) == {"DummyClassifier", "_FailingClassifier"}

    failing_row = saved_df.loc[saved_df["model_id"] == "_FailingClassifier"].iloc[0]
    assert pd.isna(failing_row["means.accuracy_score"])


def test_rerun_retries_failed_pair_after_partial_success(
    tmp_path, cv_splitter_classification
):
    """Checkpoint skips successful pairs but retries pairs that previously failed."""
    results_file = tmp_path / "results.csv"

    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )

    benchmark.run(results_file)
    assert len(benchmark.failed_experiments) == 1

    benchmark2 = ClassificationBenchmark()
    benchmark2.add_estimator(_FailingClassifier())
    benchmark2.add_estimator(DummyClassifier())
    benchmark2.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )

    results_df = benchmark2.run(results_file)

    assert len(results_df) == 2
    assert len(benchmark2.failed_experiments) == 1


def test_all_pairs_fail_returns_nan_results(
    tmp_path, cv_splitter_classification, caplog
):
    """When every pair fails, results contain NaN rows with expected columns."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )

    with caplog.at_level(logging.WARNING):
        results_df = benchmark.run(tmp_path / "results.csv")

    assert len(results_df) == 1
    assert results_df["model_id"].iloc[0] == "_FailingClassifier"
    assert pd.isna(results_df["accuracy_score_mean"].iloc[0])
    assert pd.isna(results_df["fit_time_mean"].iloc[0])
    assert "validation_id" in results_df.columns
    assert "model_id" in results_df.columns
    assert len(benchmark.failed_experiments) == 1
    assert "Benchmark completed with 1 failed task-estimator pair(s)" in caplog.text


def test_failure_logged_at_error_level(tmp_path, cv_splitter_classification, caplog):
    """Each failure is logged immediately at error level."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(_FailingClassifier())
    benchmark.add_task(
        make_classification_problem, cv_splitter_classification, [accuracy_score]
    )

    with caplog.at_level(logging.ERROR):
        benchmark.run(tmp_path / "results.csv")

    assert "Validation failed:" in caplog.text
    assert "RuntimeError: classifier fit failed" in caplog.text


def test_no_failures_no_summary_logged(tmp_path, caplog):
    """When all pairs succeed, no failure summary is logged."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_task(make_classification_problem, KFold(n_splits=3), [accuracy_score])

    with caplog.at_level(logging.WARNING):
        results_df = benchmark.run(tmp_path / "results.csv")

    assert len(results_df) == 1
    assert benchmark.failed_experiments == []
    assert "Benchmark completed with" not in caplog.text
