"""InceptionTime deep learning classifiers.

This subpackage provides InceptionTime based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "InceptionTimeClassifier",
    "InceptionTimeClassifierTorch",
]

from sktime.classification.deep_learning.inceptiontime._inceptiontime_tf import (
    InceptionTimeClassifier,
)
from sktime.classification.deep_learning.inceptiontime._inceptiontime_torch import (
    InceptionTimeClassifierTorch,
)
