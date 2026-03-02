"""Backward-compatibility wrapper for PyTorch deep classifier base class.

`BaseDeepClassifierPytorch` has been moved to
`sktime.classification.deep_learning.base._base_torch`.
This module re-exports the unified implementation to avoid code duplication.
"""

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch

__all__ = ["BaseDeepClassifierPytorch"]
