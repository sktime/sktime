"""MLP (Multi-Layer Perceptron) Classifier for Time Series Classification.

This subpackage provides Multi-Layer Perceptron (MLP) based time series
classifier in TensorFlow and PyTorch backends.
"""

__all__ = [
    "MLPClassifier",
    "MLPClassifierTorch",
]
from sktime.classification.deep_learning.mlp._mlp_tf import MLPClassifier
from sktime.classification.deep_learning.mlp._mlp_torch import MLPClassifierTorch
