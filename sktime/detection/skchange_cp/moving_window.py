"""Deprecated. Please use ``from sktime.detection import MovingWindow`` instead.

This module is kept for backward compatibility and will be removed in a future
release. The MovingWindow algorithm has been natively merged into sktime.
"""

from sktime.detection._moving_window import MovingWindow

__all__ = ["MovingWindow"]
