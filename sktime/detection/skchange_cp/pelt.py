"""Deprecated. Please use ``from sktime.detection import PELT`` instead.

This module is kept for backward compatibility and will be removed in a future
release. The PELT algorithm has been natively merged into sktime.
"""

from sktime.detection._pelt import PELT

__all__ = ["PELT"]
