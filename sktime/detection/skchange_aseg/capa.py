"""Compatibility wrapper for CAPA."""

from sktime.detection._skchange.anomaly_detectors import CAPA as _CAPA


class CAPA(_CAPA):
    """Collective and point anomaly detection."""
