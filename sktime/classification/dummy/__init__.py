"""Dummy classifier."""

__all__ = [
    "DummyClassifier",
    "DummyPretrainClassifier",
]

from sktime.classification.dummy._dummy import DummyClassifier
from sktime.classification.dummy._dummy_pretrain import DummyPretrainClassifier
