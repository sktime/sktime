import pytest

from sktime.datasets.forecasting import Airline, Longley, Lynx
from sktime.utils.estimator_checks import check_estimator

FORECASTING_DATASETS = [Airline, Longley, Lynx]


@pytest.mark.parametrize("dataset_class", FORECASTING_DATASETS)
def test_forecasting_datasets(dataset_class):
    dataset = dataset_class()
    X, y = dataset.load("X", "y")
    return 1


@pytest.mark.parametrize("dataset_class", FORECASTING_DATASETS)
def test_test(dataset_class):
    dataset = dataset_class()
    check_estimator(dataset)
