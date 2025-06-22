"""Tests for Classification Benchmark."""

__author__ = ["jgyasu"]

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import KFold

from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.datasets import load_unit_test
from sktime.utils._testing.panel import make_classification_problem


def test_classification_benchmark(tmp_path):
    """Test classification benchmark with single estimator and task."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier())
    scorers = [accuracy_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_classification_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

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


def test_add_list_estimators(tmp_path):
    """Test adding list of estimators."""
    benchmark = ClassificationBenchmark()

    estimators = [DummyClassifier(), KNeighborsTimeSeriesClassifier()]

    benchmark.add_estimator(estimators)
    scorers = [accuracy_score, brier_score_loss]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_classification_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(
            ["DummyClassifier", "KNeighborsTimeSeriesClassifier"], name="model_id"
        ),
        results_df["model_id"],
    )


def test_add_dict_estimators(tmp_path):
    """Test adding dict of estimators."""
    benchmark = ClassificationBenchmark()

    estimators = {"D": DummyClassifier(), "KN": KNeighborsTimeSeriesClassifier()}

    benchmark.add_estimator(estimators)
    scorers = [accuracy_score, brier_score_loss]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_classification_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["D", "KN"], name="model_id"),
        results_df["model_id"],
    )


def test_add_estimator_twice(tmp_path):
    """Test adding the same estimator twice."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier())
    benchmark.add_estimator(DummyClassifier())
    scorers = [accuracy_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_classification_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["DummyClassifier", "DummyClassifier_2"], name="model_id"),
        results_df["model_id"],
    )

    msg = "add_estimator does not register all estimators."
    assert len(benchmark.estimators.entities) == 2, msg


def test_add_multiple_task(tmp_path):
    """Test adding multiple tasks for benchmarking."""
    benchmark = ClassificationBenchmark()
    benchmark.add_estimator(DummyClassifier())

    dataset_loaders = [make_classification_problem, load_unit_test]
    cv_splitter = KFold(n_splits=3)
    scorers = [accuracy_score, brier_score_loss]

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
                "[dataset=make_classification_problem]_[cv_splitter=KFold]",
                "[dataset=load_unit_test]_[cv_splitter=KFold]",
            ],
            name="validation_id",
        ),
        results_df["validation_id"],
    )
