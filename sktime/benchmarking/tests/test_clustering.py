"""Tests for Clustering Benchmark."""

__author__ = ["Nischal1425"]

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.model_selection import KFold

from sktime.benchmarking.clustering import ClusteringBenchmark
from sktime.clustering.base import BaseClusterer
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.panel import make_clustering_problem


class _DummyClusterer(BaseClusterer):
    """Minimal clusterer for benchmark testing."""

    _tags = {
        "capability:out_of_sample": True,
        "capability:predict": True,
    }

    def __init__(self, n_clusters=3, random_state=42):
        self.random_state = random_state
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        self.labels_ = rng.randint(0, self.n_clusters, size=X.shape[0])
        return self

    def _predict(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        return rng.randint(0, self.n_clusters, size=X.shape[0])


class _DummyClusterer2(BaseClusterer):
    """Second minimal clusterer for multi-estimator testing."""

    _tags = {
        "capability:out_of_sample": True,
        "capability:predict": True,
    }

    def __init__(self, n_clusters=2, random_state=123):
        self.random_state = random_state
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        self.labels_ = rng.randint(0, self.n_clusters, size=X.shape[0])
        return self

    def _predict(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        return rng.randint(0, self.n_clusters, size=X.shape[0])


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("write_file", [True, False])
def test_clustering_benchmark(tmp_path, write_file):
    """Test clustering benchmark with single estimator and task."""
    benchmark = ClusteringBenchmark()
    benchmark.add_estimator(_DummyClusterer())
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    if write_file:
        results_file = tmp_path / "results.csv"
    else:
        results_file = None
    results_df = benchmark.run(results_file)

    expected_benchmark_labels = [
        "silhouette_score_fold_0_test",
        "silhouette_score_fold_1_test",
        "silhouette_score_fold_2_test",
        "silhouette_score_mean",
        "silhouette_score_std",
        "fit_time_fold_0_test",
        "fit_time_fold_1_test",
        "fit_time_fold_2_test",
        "fit_time_mean",
        "fit_time_std",
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
    benchmark = ClusteringBenchmark()

    estimators = [_DummyClusterer(), _DummyClusterer2()]

    benchmark.add_estimator(estimators)
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["_DummyClusterer", "_DummyClusterer2"], name="model_id"),
        results_df["model_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_dict_estimators(tmp_path):
    """Test adding dict of estimators."""
    benchmark = ClusteringBenchmark()

    estimators = {"DC1": _DummyClusterer(), "DC2": _DummyClusterer2()}

    benchmark.add_estimator(estimators)
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["DC1", "DC2"], name="model_id"),
        results_df["model_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_estimator_twice(tmp_path):
    """Test adding the same estimator twice."""
    benchmark = ClusteringBenchmark()
    benchmark.add_estimator(_DummyClusterer())
    benchmark.add_estimator(_DummyClusterer())
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(["_DummyClusterer", "_DummyClusterer_2"], name="model_id"),
        results_df["model_id"],
    )

    msg = "add_estimator does not register all estimators."
    assert len(benchmark.estimators.entities) == 2, msg


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_multiple_metrics(tmp_path):
    """Test benchmark with multiple metrics."""
    benchmark = ClusteringBenchmark()
    benchmark.add_estimator(_DummyClusterer())
    scorers = [silhouette_score, calinski_harabasz_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    result_rows = results_df.T.index.to_list()
    assert "silhouette_score_mean" in result_rows
    assert "calinski_harabasz_score_mean" in result_rows


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_add_multiple_tasks(tmp_path):
    """Test adding multiple tasks for benchmarking."""
    benchmark = ClusteringBenchmark()
    benchmark.add_estimator(_DummyClusterer())

    cv_splitter = KFold(n_splits=3)
    scorers = [silhouette_score]

    # Two different dataset loaders
    benchmark.add_task(
        make_clustering_problem,
        cv_splitter,
        scorers,
    )
    benchmark.add_task(
        lambda: make_clustering_problem(n_instances=30, random_state=99),
        cv_splitter,
        scorers,
        task_id="[dataset=custom_data]_[cv_splitter=KFold]",
    )

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(
            [
                "[dataset=make_clustering_problem]_[cv_splitter=KFold]",
                "[dataset=custom_data]_[cv_splitter=KFold]",
            ],
            name="validation_id",
        ),
        results_df["validation_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_custom_task_id(tmp_path):
    """Test custom task_id appears in results."""
    benchmark = ClusteringBenchmark()
    benchmark.add_estimator(_DummyClusterer())
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(
        make_clustering_problem,
        cv_splitter,
        scorers,
        task_id="my_custom_clustering_task",
    )

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    assert results_df["validation_id"].iloc[0] == "my_custom_clustering_task"


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_return_data(tmp_path):
    """Test return_data=True runs without error."""
    benchmark = ClusteringBenchmark(return_data=True)
    benchmark.add_estimator(_DummyClusterer())
    scorers = [silhouette_score]

    cv_splitter = KFold(n_splits=3)
    benchmark.add_task(make_clustering_problem, cv_splitter, scorers)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 1
