"""Compatibility wrapper for PELT."""

from sktime.detection._skchange.change_detectors import PELT as _PELT


class PELT(_PELT):
    """Pruned exact linear time changepoint detection."""
