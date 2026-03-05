"""CNTC Deep Learning Classifier.

This submodule provides the CNTC deep learning classifier
implemented in both TensorFlow and PyTorch backends.
"""

__all__ = [
    "CNTCClassifier",
    "CNTCClassifierTorch",
]

from sktime.classification.deep_learning.cntc._cntc_tf import CNTCClassifier
from sktime.classification.deep_learning.cntc._cntc_torch import CNTCClassifierTorch
