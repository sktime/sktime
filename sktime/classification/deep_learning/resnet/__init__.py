"""ResNet deep learning classifiers.

This subpackage provides ResNet based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "ResNetClassifier",
    "ResNetClassifierTorch",
]

from sktime.classification.deep_learning.resnet._resnet_tf import ResNetClassifier
from sktime.classification.deep_learning.resnet._resnet_torch import (
    ResNetClassifierTorch,
)
