"""Compatibility wrapper for MovingWindow."""

from sktime.detection._skchange.change_detectors import MovingWindow as _MovingWindow


class MovingWindow(_MovingWindow):
    """Moving window algorithm for multiple changepoint detection."""
