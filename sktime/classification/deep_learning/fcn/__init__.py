"""FCN deep learning classifiers.

This subpackage provides FCN based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "FCNClassifier",
    "FCNClassifierTorch",
]

from sktime.classification.deep_learning.fcn._fcn_tf import FCNClassifier
from sktime.classification.deep_learning.fcn._fcn_torch import FCNClassifierTorch
