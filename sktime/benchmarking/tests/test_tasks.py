__author__ = "Markus Löning"

import pytest
from pytest import raises
from sktime.benchmarking.tasks import BaseTask
from sktime.benchmarking.tasks import TSCTask
from sktime.benchmarking.tasks import TSRTask
from sktime.datasets import load_gunpoint
from sktime.datasets import load_shampoo_sales

TASKS = (TSCTask, TSRTask)

gunpoint = load_gunpoint(return_X_y=False)
shampoo_sales = load_shampoo_sales()

BASE_READONLY_ATTRS = ("target", "features", "metadata")


# Test read-only attributes of base task
@pytest.mark.parametrize("attr", BASE_READONLY_ATTRS)
def test_readonly_attributes(attr):
    task = BaseTask(target='class_val', metadata=gunpoint)
    with raises(AttributeError):
        task.__setattr__(attr, "val")


# Test data compatibility checks
@pytest.mark.parametrize("task", TASKS)
def test_check_data_compatibility(task):
    task = task(target='target')
    with raises(ValueError):
        task.set_metadata(gunpoint)


# Test setting of metadata
def check_set_metadata(task, target, metadata):
    task = task(target=target)
    assert task.metadata is None

    task.set_metadata(metadata)
    assert task.metadata is not None

    # cannot be re-set
    with raises(AttributeError):
        task.set_metadata(metadata)


@pytest.mark.parametrize("task", [TSRTask, TSCTask])
def test_set_metadata_supervised(task):
    check_set_metadata(task, 'class_val', gunpoint)
