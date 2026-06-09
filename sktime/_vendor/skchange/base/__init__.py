"""Base classes for detectors and other objects in skchange."""

from ._base_detector import BaseDetector
from ._base_interval_scorer import BaseIntervalScorer

__all__ = ["BaseDetector", "BaseIntervalScorer"]
