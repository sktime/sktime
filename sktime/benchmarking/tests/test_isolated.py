"""Tests for isolated benchmark execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sktime
from sktime.benchmarking._benchmarking_dataclasses import TaskObject
from sktime.benchmarking._isolated_requirements import collect_pair_requirements
from sktime.benchmarking.benchmarks import BaseBenchmark
from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.benchmarking.regression import RegressionBenchmark
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._env_managers import BaseEnvironmentManager, UvEnvironmentManager

__author__ = ["jgyasu"]


class _Tagged:
    """Minimal object that exposes a ``python_dependencies`` tag."""

    def __init__(self, deps):
        self._deps = deps

    def get_class_tag(self, tag):
        if tag == "python_dependencies":
            return self._deps
        return None


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_collect_pair_requirements_unions_tags():
    """Estimator, scorer, splitter, and dataset tags are de-duplicated."""
    task = TaskObject(
        data=_Tagged(["dataset_pkg", "shared_pkg"]),
        cv_splitter=_Tagged(["splitter_pkg"]),
        scorers=[_Tagged(["scorer_pkg", "shared_pkg"])],
    )
    estimator = _Tagged(["estimator_pkg", "shared_pkg"])

    assert collect_pair_requirements(estimator, task) == [
        "dataset_pkg",
        "estimator_pkg",
        "scorer_pkg",
        "shared_pkg",
        "splitter_pkg",
    ]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_run_isolated_validation_uses_env_manager_run():
    """The isolated runner launches validation through ``env_manager.run``."""
    cloudpickle = pytest.importorskip("cloudpickle")

    from sktime.benchmarking._isolated_runner import (
        _execute_validation,
        run_isolated_validation,
    )

    captured = {}

    class _RecordingManager(BaseEnvironmentManager):
        def get_python_executable(self, requirements=None, python=None):
            return Path(sys.executable)

        def run(self, target, *, requirements=None, args=None, **kwargs):
            captured["target"] = target
            captured["requirements"] = list(requirements or [])
            captured["args"] = args
            return SimpleNamespace(
                returncode=0,
                stdout=cloudpickle.dumps({"status": "success", "folds": {"0": 1}}),
                stderr=b"",
            )

    task = TaskObject(
        data=_Tagged(["pandas"]),
        cv_splitter=_Tagged(None),
        scorers=[_Tagged(None)],
    )
    estimator = _Tagged(["numpy"])
    manager = _RecordingManager()
    init_params = {"backend": "loky", "env_mode": "current"}

    folds = run_isolated_validation(
        benchmark_cls=ForecastingBenchmark,
        init_params=init_params,
        task=task,
        estimator=estimator,
        env_manager=manager,
    )

    assert folds == {"0": 1}
    assert captured["target"] is _execute_validation
    assert captured["requirements"] == ["numpy", "pandas"]
    assert captured["args"][0] is ForecastingBenchmark
    assert captured["args"][1] == init_params
    assert captured["args"][3]._deps == estimator._deps


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_get_env_manager_uses_utils_uv_manager(tmp_path):
    """The default isolated manager is the shared ``UvEnvironmentManager``."""
    benchmark = ForecastingBenchmark(
        env_mode="per_task_estimator",
        envs_dir=tmp_path,
    )
    manager = benchmark._get_env_manager()

    assert isinstance(manager, UvEnvironmentManager)
    assert manager.envs_dir == tmp_path
    assert manager.python == sys.executable
    assert manager.base_requirements == [
        "cloudpickle",
        f"sktime=={sktime.__version__}",
    ]
    assert manager.editable == []


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_passed_env_manager_implies_isolated():
    """Passing ``env_manager`` enables isolated execution and is reused."""
    manager = object()
    benchmark = ForecastingBenchmark(env_manager=manager)

    assert benchmark._isolated is True
    assert benchmark._get_env_manager() is manager


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize(
    "benchmark_cls",
    [ForecastingBenchmark, ClassificationBenchmark, RegressionBenchmark],
)
def test_isolated_validation_passes_benchmark_class(benchmark_cls):
    """Isolation sends the benchmark class, not a kind string."""
    cloudpickle = pytest.importorskip("cloudpickle")

    captured = {}

    class _RecordingManager(BaseEnvironmentManager):
        def get_python_executable(self, requirements=None, python=None):
            return Path(sys.executable)

        def run(self, target, *, requirements=None, args=None, **kwargs):
            captured["args"] = args
            return SimpleNamespace(
                returncode=0,
                stdout=cloudpickle.dumps({"status": "success", "folds": {}}),
                stderr=b"",
            )

    task = TaskObject(
        data=_Tagged(None),
        cv_splitter=_Tagged(None),
        scorers=[_Tagged(None)],
    )
    benchmark = benchmark_cls(
        backend="loky",
        return_data=True,
        env_mode="per_task_estimator",
        env_manager=_RecordingManager(),
    )
    benchmark._run_validation_isolated(task, _Tagged(None))

    assert captured["args"][0] is benchmark_cls
    assert captured["args"][1] == {
        "id_format": None,
        "backend": "loky",
        "backend_params": None,
        "return_data": True,
        "env_mode": "current",
    }


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_isolated_validation_calls_subclass_on_fresh_instance():
    """A subclass with no kind tag runs its own ``_run_validation``.

    The child is a new instance, so estimators registered on the parent
    are not present.
    """
    cloudpickle = pytest.importorskip("cloudpickle")

    class _MarkerBenchmark(BaseBenchmark):
        def _run_validation(self, task, estimator):
            return {
                "marker": type(self).__name__,
                "backend": self.backend,
                "n_estimators": len(self.estimators.entities),
                "estimator": estimator,
            }

    class _RecordingManager(BaseEnvironmentManager):
        def get_python_executable(self, requirements=None, python=None):
            return Path(sys.executable)

        def run(self, target, *, requirements=None, args=None, **kwargs):
            result = target(*args)
            return SimpleNamespace(
                returncode=0,
                stdout=cloudpickle.dumps(result),
                stderr=b"",
            )

    task = TaskObject(
        data=_Tagged(None),
        cv_splitter=_Tagged(None),
        scorers=[_Tagged(None)],
    )
    estimator = _Tagged(["numpy"])
    benchmark = _MarkerBenchmark(
        backend="loky",
        env_mode="per_task_estimator",
        env_manager=_RecordingManager(),
    )
    benchmark.estimators.entities["kept-on-parent"] = estimator

    folds = benchmark._run_validation_isolated(task, estimator)

    assert folds["marker"] == "_MarkerBenchmark"
    assert folds["backend"] == "loky"
    assert folds["n_estimators"] == 0
    assert folds["estimator"]._deps == ["numpy"]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_isolated_validation_reraises_builtin_errors():
    """Errors from ``_run_validation`` are raised again in the parent."""
    cloudpickle = pytest.importorskip("cloudpickle")

    from sktime.benchmarking._isolated_runner import run_isolated_validation

    class _BoomBenchmark(BaseBenchmark):
        def _run_validation(self, task, estimator):
            raise ValueError("fold failed")

    class _RecordingManager(BaseEnvironmentManager):
        def get_python_executable(self, requirements=None, python=None):
            return Path(sys.executable)

        def run(self, target, *, requirements=None, args=None, **kwargs):
            result = target(*args)
            return SimpleNamespace(
                returncode=0,
                stdout=cloudpickle.dumps(result),
                stderr=b"",
            )

    task = TaskObject(
        data=_Tagged(None),
        cv_splitter=_Tagged(None),
        scorers=[_Tagged(None)],
    )
    with pytest.raises(ValueError, match="fold failed"):
        run_isolated_validation(
            benchmark_cls=_BoomBenchmark,
            init_params={},
            task=task,
            estimator=_Tagged(None),
            env_manager=_RecordingManager(),
        )
