"""Classification foundation models."""

__all__ = [
    "MantisClassifier",
    "MomentFMClassifier",
    "TSPulseClassifier",
]

from sktime.classification.foundation_models.mantis import MantisClassifier
from sktime.classification.foundation_models.momentfm import MomentFMClassifier
from sktime.classification.foundation_models.tspulse import TSPulseClassifier
