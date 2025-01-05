"""Classifier Base."""

__all__ = [
    "BaseClassifier",
    "OSCNNClassifier"
]

from sktime.classification.base import BaseClassifier
from sktime.classification.oscnn import OSCNNClassifier