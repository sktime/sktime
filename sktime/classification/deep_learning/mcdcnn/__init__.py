"""MCDCNN deep learning classifiers.

This subpackage provides MCDCNN based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MCDCNNClassifier",
    "MCDCNNClassifierTorch",
]

from sktime.classification.deep_learning.mcdcnn._mcdcnn_tf import MCDCNNClassifier
from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import MCDCNNClassifierTorch
