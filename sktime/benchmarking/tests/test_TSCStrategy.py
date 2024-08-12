"""Data storage for benchmarking."""

import pytest

from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
from sktime.datasets import load_gunpoint, load_italy_power_demand
from sktime.tests.test_switch import run_test_module_changed

classifier = ComposableTimeSeriesForestClassifier(n_estimators=2)

DATASET_LOADERS = (load_gunpoint, load_italy_power_demand)


# Test output of time-series classification strategies
@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("dataset", DATASET_LOADERS)
def test_TSCStrategy(dataset):
    """Test strategy."""
    train = dataset(split="train", return_X_y=False)
    test = dataset(split="test", return_X_y=False)
    s = TSCStrategy(classifier)
    task = TSCTask(target="class_val")
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape
