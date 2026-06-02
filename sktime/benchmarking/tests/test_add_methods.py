"""Tests for Different Ways to Add Tasks."""

__author__ = ["jgyasu"]

import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.classification.dummy import DummyClassifier
from sktime.datasets import ArrowHead
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
class TestBenchmarkAddMethods:
    """Test benchmark.add() method with different input formats."""

    def _assert_standard_metrics(self, results_df):
        """Helper to check standard benchmark outputs."""
        expected = [
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

        rows = results_df.T.index.to_list()
        for metric in expected:
            assert metric in rows, f"{metric} not found in result rows"

    def test_add_task_tuple(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        cv_splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add((dataset, scorer, cv_splitter))
        results_df = benchmark.run()

        self._assert_standard_metrics(results_df)

    def test_add_task_individually(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add(dataset)
        benchmark.add(scorer)
        benchmark.add(splitter)

        results_df = benchmark.run()
        self._assert_standard_metrics(results_df)

    def test_add_task_tuple_order_invariant(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add((scorer, dataset, splitter))  # shuffled
        results_df = benchmark.run()

        self._assert_standard_metrics(results_df)

    def test_add_task_tuple_missing_component(self):
        benchmark = ClassificationBenchmark()

        scorer = accuracy_score
        dataset = ArrowHead()

        with pytest.raises(TypeError, match="Unsupported tuple format"):
            benchmark.add((dataset, scorer))

    def test_add_task_tuple_duplicate_roles(self):
        benchmark = ClassificationBenchmark()

        dataset1 = ArrowHead()
        dataset2 = ArrowHead()
        splitter = KFold(n_splits=3)

        with pytest.raises(TypeError, match="Multiple datasets provided in tuple"):
            benchmark.add((dataset1, dataset2, splitter))

    def test_add_task_tuple_invalid_object(self):
        benchmark = ClassificationBenchmark()

        dataset = ArrowHead()
        splitter = KFold(n_splits=3)

        with pytest.raises(TypeError):
            benchmark.add((dataset, "not_a_metric", splitter))

    def test_add_deduplicates_components(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add(dataset)
        benchmark.add(dataset)
        benchmark.add(scorer)
        benchmark.add(scorer)
        benchmark.add(splitter)
        benchmark.add(splitter)

        results_df = benchmark.run()
        self._assert_standard_metrics(results_df)

    def test_add_mixed_inputs(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        # invalid mixed tuple length
        with pytest.raises(TypeError, match="Unsupported tuple format"):
            benchmark.add((scorer, splitter))

        # valid flow
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())
        benchmark.add(dataset)
        benchmark.add(scorer)
        benchmark.add(splitter)

        results_df = benchmark.run()
        self._assert_standard_metrics(results_df)

    def test_add_multiple_tasks(self):
        benchmark = ClassificationBenchmark()
        benchmark.add(DummyClassifier())

        scorer = accuracy_score
        splitter = KFold(n_splits=3)

        dataset1 = ArrowHead()
        dataset2 = ArrowHead()

        benchmark.add(
            (dataset1, scorer, splitter),
            (scorer, dataset2, splitter),
        )

        results_df = benchmark.run()
        self._assert_standard_metrics(results_df)

    def test_add_estimator_dict(self):
        """Test adding estimators using a dictionary for custom IDs."""
        benchmark = ClassificationBenchmark()

        clf = DummyClassifier()
        benchmark.add({"dummy_custom_id": clf})

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add((dataset, scorer, splitter))
        results_df = benchmark.run()

        # Check that the custom ID from the dictionary was used
        assert results_df.loc[0, "model_id"] == "dummy_custom_id"

    def test_add_estimator_list(self):
        """Test adding multiple estimators using a list."""
        benchmark = ClassificationBenchmark()

        clfs = [DummyClassifier()]
        benchmark.add(clfs)

        scorer = accuracy_score
        splitter = KFold(n_splits=3)
        dataset = ArrowHead()

        benchmark.add((dataset, scorer, splitter))
        results_df = benchmark.run()

        # Check that the default class name was generated as the ID
        assert results_df.loc[0, "model_id"] == "DummyClassifier"

    def test_unsupported_tuple_length(self):
        """Test that passing length-2 tuples (old estimator/ID behavior) fails."""
        benchmark = ClassificationBenchmark()
        clf = DummyClassifier()

        with pytest.raises(TypeError, match="Unsupported tuple format of length 2"):
            benchmark.add((clf, "should_fail"))
