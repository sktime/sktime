"""Compatibility wrapper for SeededBinarySegmentation."""

from sktime.detection._skchange.change_detectors import (
    SeededBinarySegmentation as _SeededBinarySegmentation,
)


class SeededBinarySegmentation(_SeededBinarySegmentation):
    """Seeded binary segmentation algorithm for changepoint detection."""
