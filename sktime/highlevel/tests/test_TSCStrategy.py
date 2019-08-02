import pytest

from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.datasets import load_gunpoint
from sktime.datasets import load_italy_power_demand
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask

classifier = TimeSeriesForestClassifier(n_estimators=2)

DATASET_LOADERS = (load_gunpoint, load_italy_power_demand)


# Test output of time-series classification strategies
@pytest.mark.parametrize("dataset", DATASET_LOADERS)
def test_TSCStrategy(dataset):
    train = dataset(split='TRAIN')
    test = dataset(split='TEST')
    s = TSCStrategy(classifier)
    task = TSCTask(target='class_val')
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape
