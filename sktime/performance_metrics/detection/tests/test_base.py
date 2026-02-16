"""Tests for base class boilerplate."""

import pandas as pd
import pytest

from sktime.detection._datatypes._examples import (
    _get_example_points_2,
    _get_example_points_3,
    _get_example_segments_2,
    _get_example_segments_3,
)
from sktime.performance_metrics.detection._chamfer import DirectedChamfer
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.performance_metrics.detection"),
    reason="run test only if detection module changed",
)
def test_input_coercions():
    """Test that segments are coerced for point metrics."""
    y_pred_seg = _get_example_segments_2()
    y_true_seg = _get_example_segments_3()

    y_pred_pts = _get_example_points_2()
    y_true_pts = _get_example_points_3()

    metric = DirectedChamfer()

    loss_seg = metric(y_true_seg, y_pred_seg)
    loss_pts = metric(y_true_pts, y_pred_pts)
    assert loss_seg == loss_pts

    X = pd.DataFrame({"range10": range(10)})

    loss_seg_X = metric(y_true_seg, y_pred_seg, X)
    loss_pts_X = metric(y_true_pts, y_pred_pts, X)
    assert loss_seg_X == loss_pts_X
