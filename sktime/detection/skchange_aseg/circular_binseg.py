"""Compatibility wrapper for CircularBinarySegmentation."""

from sktime.detection._skchange.anomaly_detectors import (
    CircularBinarySegmentation as _CircularBinarySegmentation,
)


class CircularBinarySegmentation(_CircularBinarySegmentation):
    """Circular binary segmentation algorithm for anomalous segment detection."""
