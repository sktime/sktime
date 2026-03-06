"""Tests for Different Ways to Add Tasks."""

__author__ = ["jgyasu"]

import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.datasets import (
    ArrowHead,
)
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_task_tuple():
    """Test adding tasks as tuple."""
    benchmark = ClassificationBenchmark()
    benchmark.add(DummyClassifier())
    scorer = accuracy_score

    cv_splitter = KFold(n_splits=3)
    dataset_loader = ArrowHead()
    task = (dataset_loader, scorer, cv_splitter)
    benchmark.add(task)

    results_df = benchmark.run()
    expected_benchmark_labels = [
        "accuracy_score_fold_0_test",
        "accuracy_score_fold_1_test",
        "accuracy_score_fold_2_test",
        "accuracy_score_mean",
        "accuracy_score_std",
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
def test_add_task_individually():
    """Test adding tasks individually."""
    benchmark = ClassificationBenchmark()
    benchmark.add(DummyClassifier())
    scorer = accuracy_score

    cv_splitter = KFold(n_splits=3)
    dataset_loader = ArrowHead()

    benchmark.add(dataset_loader)
    benchmark.add(scorer)
    benchmark.add(cv_splitter)

    results_df = benchmark.run()
    expected_benchmark_labels = [
        "accuracy_score_fold_0_test",
        "accuracy_score_fold_1_test",
        "accuracy_score_fold_2_test",
        "accuracy_score_mean",
        "accuracy_score_std",
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
