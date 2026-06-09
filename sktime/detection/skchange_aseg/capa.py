"""Deprecated. Please use ``from sktime.detection import CAPA`` instead.

This module is kept for backward compatibility and will be removed in a future
release. The CAPA algorithm has been natively merged into sktime.
"""

from sktime.detection._capa import CAPA

__all__ = ["CAPA"]
