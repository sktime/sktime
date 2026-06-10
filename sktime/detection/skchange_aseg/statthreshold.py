"""Deprecated. Use ``from sktime.detection import StatThresholdAnomaliser``.

This module is kept for backward compatibility and will be removed in a future
release. StatThresholdAnomaliser has been natively merged into sktime.
"""

from sktime.detection._stat_threshold_anomaliser import StatThresholdAnomaliser

__all__ = ["StatThresholdAnomaliser"]
