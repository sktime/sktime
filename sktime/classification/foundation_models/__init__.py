"""Classification foundation models."""

__all__ = [
    "MomentFMClassifier",
    "TSPulseClassifier",
]

from sktime.classification.foundation_models.momentfm import MomentFMClassifier
from sktime.classification.foundation_models.tspulse import TSPulseClassifier
