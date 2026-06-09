"""Compatibility wrapper for StatThresholdAnomaliser."""

from sktime.detection._skchange.anomaly_detectors import (
    StatThresholdAnomaliser as _StatThresholdAnomaliser,
)


class StatThresholdAnomaliser(_StatThresholdAnomaliser):
    """Anomaly detection based on thresholding the values of segment statistics.

    Parameters
    ----------
    change_detector : BaseChangeDetector
        Change detector to use for detecting segments.
    stat : callable, optional (default=np.mean)
        Statistic to calculate per segment. A function that takes in a 1D array and
        returns a float.
    stat_lower : float, optional (default=-1.0)
        Segments with a statistic lower than this value are considered anomalous.
    stat_upper : float, optional (default=1.0)
        Segments with a statistic higher than this value are considered anomalous.
    """
