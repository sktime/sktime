"""Tests for add API in benchmarking."""

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
class TestBenchmarkAddAPI:
    """Tests for different ways of adding items to a benchmark."""

    @staticmethod
    def _make_components():
        return {
            "estimator": DummyClassifier(),
            "dataset": ArrowHead(),
            "metric": accuracy_score,
            "cv": KFold(n_splits=3),
        }

    def _run_benchmark(self, benchmark):
        results = benchmark.run()
        assert not results.empty
        return results

    def _assert_basic_structure(self, results):
        """Assert core result schema is present."""
        assert "validation_id" in results.columns
        assert "model_id" in results.columns

    def _assert_metrics_present(self, results):
        """Assert at least one aggregated metric exists."""
        assert any(col.endswith("_mean") for col in results.columns)

    def test_task_tuple_equivalent_to_individual_add(self):
        comps = self._make_components()

        bench_tuple = ClassificationBenchmark()
        bench_individual = ClassificationBenchmark()

        bench_tuple.add(comps["estimator"])
        bench_tuple.add((comps["dataset"], comps["metric"], comps["cv"]))

        bench_individual.add(comps["estimator"])
        bench_individual.add(comps["dataset"])
        bench_individual.add(comps["metric"])
        bench_individual.add(comps["cv"])

        res_tuple = self._run_benchmark(bench_tuple)
        res_individual = self._run_benchmark(bench_individual)

        self._assert_basic_structure(res_tuple)
        self._assert_basic_structure(res_individual)

        # Same tasks should be run
        assert set(res_tuple["validation_id"]) == set(res_individual["validation_id"])

        # Same estimators should be present
        assert set(res_tuple["model_id"]) == set(res_individual["model_id"])

    def test_add_named_estimator(self):
        comps = self._make_components()

        benchmark = ClassificationBenchmark()
        benchmark.add(("dummy_named", comps["estimator"]))
        benchmark.add(comps["dataset"], comps["metric"], comps["cv"])

        results = self._run_benchmark(benchmark)

        self._assert_basic_structure(results)

        assert "dummy_named" in results["model_id"].values

    @pytest.mark.parametrize(
        "add_order",
        [
            ("estimator", "dataset", "metric", "cv"),
            ("dataset", "estimator", "cv", "metric"),
            ("metric", "cv", "dataset", "estimator"),
        ],
    )
    def test_mixed_add_order(self, add_order):
        comps = self._make_components()
        benchmark = ClassificationBenchmark()

        for key in add_order:
            benchmark.add(comps[key])

        results = self._run_benchmark(benchmark)

        self._assert_basic_structure(results)
        self._assert_metrics_present(results)

    def test_unknown_object_type_raises(self):
        benchmark = ClassificationBenchmark()

        with pytest.raises(TypeError):
            benchmark.add(object())
