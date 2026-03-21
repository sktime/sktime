"""Compatibility wrapper for StatThresholdAnomaliser."""

from sktime.detection._skchange.anomaly_detectors import (
    StatThresholdAnomaliser as _StatThresholdAnomaliser,
)


class StatThresholdAnomaliser(_StatThresholdAnomaliser):
    """Anomaly detection based on thresholding values of segment statistics."""
