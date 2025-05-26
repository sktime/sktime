"""Metrics for detection tasks."""

from sktime.performance_metrics.detection._chamfer import DirectedChamfer
from sktime.performance_metrics.detection._count import DetectionCount
from sktime.performance_metrics.detection._f1score import WindowedF1Score
from sktime.performance_metrics.detection._hausdorff import DirectedHausdorff
from sktime.performance_metrics.detection._randindex import RandIndex
from sktime.performance_metrics.detection._ts_auprc import TimeSeriesAUPRC

# TimeSeriesAUPRC is a utility for computing AUPRC, not strictly a detection metric,
# but is included here as it’s used in the test suite and utilities for detection metrics.
__all__ = [
    "DirectedChamfer",
    "DirectedHausdorff",
    "DetectionCount",
    "WindowedF1Score",
    "RandIndex",
    "TimeSeriesAUPRC",
]
