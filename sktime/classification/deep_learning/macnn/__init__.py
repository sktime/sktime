"""MACNN deep learning classifiers.

This subpackage provides MACNN based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MACNNClassifier",
    "MACNNClassifierTorch",
]

from sktime.classification.deep_learning.macnn._macnn_tf import MACNNClassifier
from sktime.classification.deep_learning.macnn._macnn_torch import MACNNClassifierTorch
