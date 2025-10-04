"""CNN classifiers for time series classification."""

from sktime.classification.deep_learning.cnn._cnn_tf import CNNClassifier
from sktime.classification.deep_learning.cnn._cnn_torch import CNNClassifierTorch

__all__ = ["CNNClassifier", "CNNClassifierTorch"]
