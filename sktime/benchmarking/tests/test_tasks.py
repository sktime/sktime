"""Tests for benchmarking tasks."""

__author__ = ["mloning"]

import pytest
from pytest import raises

from sktime.benchmarking.tasks import BaseTask, TSCTask, TSRTask
from sktime.datasets import load_gunpoint, load_shampoo_sales
from sktime.tests.test_switch import run_test_module_changed

TASKS = (TSCTask, TSRTask)

gunpoint = load_gunpoint(return_X_y=False)
shampoo_sales = load_shampoo_sales()

BASE_READONLY_ATTRS = ("target", "features", "metadata")


# Test read-only attributes of base task
@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("attr", BASE_READONLY_ATTRS)
def test_readonly_attributes(attr):
    """Test read-only attributes."""
    task = BaseTask(target="class_val", metadata=gunpoint)
    with raises(AttributeError):
        task.__setattr__(attr, "val")


# Test data compatibility checks
@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("task", TASKS)
def test_check_data_compatibility(task):
    """Check data compatibility."""
    task = task(target="target")
    with raises(ValueError):
        task.set_metadata(gunpoint)


# Test setting of metadata
def check_set_metadata(task, target, metadata):
    """Check set_metadata."""
    task = task(target=target)
    assert task.metadata is None

    task.set_metadata(metadata)
    assert task.metadata is not None

    # cannot be re-set
    with raises(AttributeError):
        task.set_metadata(metadata)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("task", [TSRTask, TSCTask])
def test_set_metadata_supervised(task):
    """Test check_set_metadata."""
    check_set_metadata(task, "class_val", gunpoint)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_invalid_task_init_inputs():
    """Test constructor validation for invalid target/features inputs."""
    with raises(ValueError, match="target must be a non-empty string"):
        BaseTask(target="")

    with raises(TypeError, match="features must be a list-like of strings"):
        BaseTask(target="class_val", features="dim_0")

    with raises(ValueError, match="must not contain duplicate"):
        BaseTask(target="class_val", features=["dim_0", "dim_0"])

    with raises(ValueError, match="must not also be included in features"):
        BaseTask(target="class_val", features=["dim_0", "class_val"])
