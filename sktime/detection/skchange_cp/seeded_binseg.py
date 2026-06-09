"""Deprecated. Use ``from sktime.detection import SeededBinarySegmentation``.

This module is kept for backward compatibility and will be removed in a future
release. SeededBinarySegmentation has been natively merged into sktime.
"""

from sktime.detection._seeded_binseg import SeededBinarySegmentation

__all__ = ["SeededBinarySegmentation"]
