"""TapNet deep learning classifiers.

This subpackage provides TapNet based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "TapNetClassifier",
    "TapNetClassifierTorch",
]

from sktime.classification.deep_learning.tapnet._tapnet_tf import TapNetClassifier
from sktime.classification.deep_learning.tapnet._tapnet_torch import (
    TapNetClassifierTorch,
)
