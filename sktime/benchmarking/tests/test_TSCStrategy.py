# -*- coding: utf-8 -*-
"""Test TSCStrategy."""
import pytest

from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.compose import ComposableTimeSeriesForestClassifier
from sktime.datasets import load_gunpoint, load_italy_power_demand

classifier = ComposableTimeSeriesForestClassifier(n_estimators=2)

DATASET_LOADERS = (load_gunpoint, load_italy_power_demand)


# Test output of time-series classification strategies
@pytest.mark.parametrize("dataset", DATASET_LOADERS)
def test_TSCStrategy(dataset):
    """Test TSC Strategy."""
    train = dataset(split="train", return_X_y=False)
    test = dataset(split="test", return_X_y=False)
    s = TSCStrategy(classifier)
    task = TSCTask(target="class_val")
    s.fit(task, train)
    y_pred = s.predict(test)
    assert y_pred.shape == test[task.target].shape
