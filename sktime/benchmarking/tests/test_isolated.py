"""Tests for isolated benchmark execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sktime
from sktime.benchmarking._benchmarking_dataclasses import TaskObject
from sktime.benchmarking._isolated_requirements import collect_pair_requirements
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.env_managers import BaseEnvironmentManager, UvEnvironmentManager

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
    """The isolated runner launches the worker through ``env_manager.run``."""
    cloudpickle = pytest.importorskip("cloudpickle")

    from sktime.benchmarking._isolated_runner import run_isolated_validation

    captured = {}

    class _RecordingManager(BaseEnvironmentManager):
        def get_python_executable(self, requirements=None, python=None):
            return Path(sys.executable)

        def run(self, target, *, requirements=None, input=None, **kwargs):
            captured["target"] = target
            captured["requirements"] = list(requirements or [])
            captured["payload"] = cloudpickle.loads(input)
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

    folds = run_isolated_validation(
        benchmark_kind="forecasting",
        task=task,
        estimator=estimator,
        backend=None,
        backend_params=None,
        return_data=False,
        env_manager=manager,
    )

    assert folds == {"0": 1}
    assert captured["target"] == "sktime.benchmarking._worker"
    assert captured["requirements"] == ["numpy", "pandas"]
    assert captured["payload"]["benchmark_kind"] == "forecasting"
    assert captured["payload"]["estimator"]._deps == estimator._deps


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_get_env_manager_uses_utils_uv_manager(tmp_path):
    """The default isolated manager is the shared ``UvEnvironmentManager``."""
    benchmark = ForecastingBenchmark(
        isolated=True,
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

    assert benchmark.isolated is True
    assert benchmark._get_env_manager() is manager
