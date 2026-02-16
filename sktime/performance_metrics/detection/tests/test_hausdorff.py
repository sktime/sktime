"""Tests for the directed Hausdorff distance."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._hausdorff import DirectedHausdorff
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DirectedHausdorff),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hausdorff():
    """Test the directed Hausdorff distance."""
    y_pred = pd.DataFrame({"ilocs": [0, 1, 3, 4, 5]})  # locs are 1, 2, 5, 42, 43
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})  # locs are 1, 4, 5
    X = pd.DataFrame({"foo": [8, 4, 3, 7, 10, 12]}, index=[1, 2, 4, 5, 42, 43])

    metric = DirectedHausdorff()

    loss1 = metric(y_true, y_pred)
    assert isinstance(loss1, float)
    # closest ilocs are 0, 0, 3, 3, 3
    # absolute distance is max(0, 1, 0, 1, 2) = 2
    assert loss1 == 2

    loss2 = metric(y_true, y_pred, X)
    assert isinstance(loss2, float)
    # predicted locs are 1, 2, 5, 42, 43
    # closest locs are 1, 1, 5, 5, 5
    # absolute distance is max(0, 1, 0, 37, 38) = 38
    assert loss2 == 38
