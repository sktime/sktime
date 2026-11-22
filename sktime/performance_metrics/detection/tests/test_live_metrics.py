"""Tests shared by the live detection metrics."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._event_tpr import EventTPR
from sktime.performance_metrics.detection._false_alarm_rate import FalseAlarmRate
from sktime.performance_metrics.detection._mean_detection_offset import (
    MeanDetectionOffset,
)
from sktime.tests.test_switch import run_test_module_changed

SKIP_IF_UNCHANGED = pytest.mark.skipif(
    not run_test_module_changed("sktime.performance_metrics.detection"),
    reason="run test only if detection module changed",
)


@SKIP_IF_UNCHANGED
@pytest.mark.parametrize(
    "metric_class", [EventTPR, MeanDetectionOffset, FalseAlarmRate]
)
def test_live_metrics_refuse_interval_ground_truth(metric_class):
    """Interval ground truth raises, instead of becoming its end points.

    Without the check, the two intervals below would become the point events
    2, 5, 7 and 9. The alarms inside them would then be scored as misses or
    as false alarms, with no error.
    """
    X = pd.DataFrame({"foo": range(10)})
    intervals = pd.IntervalIndex.from_tuples([(2, 5), (7, 9)], closed="left")
    y_true = pd.DataFrame({"ilocs": intervals})
    y_pred = pd.DataFrame({"ilocs": [3, 8]}, dtype="int64")  # inside both

    with pytest.raises(ValueError, match="scores point events only"):
        metric_class()(y_true, y_pred, X)
