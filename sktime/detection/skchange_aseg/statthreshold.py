"""Anomaly detectors composed of change detectors and thresholding logic."""

from sktime.detection._skchange.anomaly_detectors import StatThresholdAnomaliser

__all__ = ["StatThresholdAnomaliser"]
