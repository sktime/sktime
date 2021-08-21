# -*- coding: utf-8 -*-
"""Feature based time series classifiers."""
__all__ = [
    "Catch22Classifier",
    "MatrixProfileClassifier",
    "SignatureClassifier",
    "TSFreshClassifier",
]

from sktime.classification.feature_based._catch22_classifier import Catch22Classifier
from sktime.classification.feature_based._matrix_profile_classifier import (
    MatrixProfileClassifier,
)
from sktime.classification.feature_based._signature_classifier import (
    SignatureClassifier,
)
from sktime.classification.feature_based._tsfresh_classifier import TSFreshClassifier
