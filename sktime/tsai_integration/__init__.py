"""
tsai_integration
================
Thin wrappers that let you call tsai’s deep-learning models from sktime’s API.
"""
from .delegated_classifier import TsaiTSTClassifier, TsaiInceptionTimeClassifier

__all__ = ["TsaiTSTClassifier", "TsaiInceptionTimeClassifier"]
