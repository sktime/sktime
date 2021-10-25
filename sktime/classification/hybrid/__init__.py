# -*- coding: utf-8 -*-
"""Hybrid time series classifiers."""
__all__ = [
    "Catch22ForestClassifier",
    "HIVECOTEV1",
    "HIVECOTEV2",
]

from sktime.classification.hybrid._catch22_forest_classifier import (
    Catch22ForestClassifier,
)
from sktime.classification.hybrid._hivecote_v1 import HIVECOTEV1
from sktime.classification.hybrid._hivecote_v2 import HIVECOTEV2
