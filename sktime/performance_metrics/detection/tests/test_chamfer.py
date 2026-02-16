"""Tests for the directed Chamfer distance."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._chamfer import DirectedChamfer
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DirectedChamfer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_chamfer():
    """Test the directed Chamfer distance."""
    y_pred = pd.DataFrame({"ilocs": [0, 1, 3, 4, 5]})  # locs are 1, 2, 5, 42, 43
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})  # locs are 1, 4, 5
    X = pd.DataFrame({"foo": [8, 4, 3, 7, 10, 12]}, index=[1, 2, 4, 5, 42, 43])

    metric = DirectedChamfer()

    loss1 = metric(y_true, y_pred)
    assert isinstance(loss1, float)
    # closest ilocs are 0, 0, 3, 3, 3
    # absolute distance is 0 + 1 + 0 + 1 + 2 = 4
    assert loss1 == 4

    loss2 = metric(y_true, y_pred, X)
    assert isinstance(loss2, float)
    # predicted locs are 1, 2, 5, 42, 43
    # closest locs are 1, 1, 5, 5, 5
    # distances are 0 + 1 + 0 + 37 + 38 = 76
    assert loss2 == 76
