import pytest
from pytest import raises

from sktime.datasets import load_gunpoint
from sktime.datasets import load_shampoo_sales
from sktime.highlevel.tasks import BaseTask
from sktime.highlevel.tasks import TSCTask
from sktime.highlevel.tasks import TSRTask
from sktime.highlevel.tasks import ForecastingTask

__author__ = "Markus Löning"

TASKS = (TSCTask, TSRTask, ForecastingTask)

gunpoint = load_gunpoint(return_X_y=False)
shampoo_sales = load_shampoo_sales(return_y_as_dataframe=True)

BASE_READONLY_ATTRS = ("target", "features", "metadata")


# Test read-only attributes of base task
@pytest.mark.parametrize("attr", BASE_READONLY_ATTRS)
def test_readonly_attributes(attr):
    task = BaseTask(target='class_val', metadata=gunpoint)
    with raises(AttributeError):
        task.__setattr__(attr, "val")


# Test read-only forecasting horizon attribute of forecasting task
@pytest.mark.parametrize("fh", [None, [1], [1, 2, 3]])
def test_readonly_fh(fh):
    task = ForecastingTask(target='ShampooSales', metadata=shampoo_sales)
    with raises(AttributeError):
        task.fh = None


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


def test_set_metadata_forecasting():
    check_set_metadata(ForecastingTask, 'ShampooSales', shampoo_sales)

