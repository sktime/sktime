"""RNN deep learning classifiers.

This subpackage provides RNN based classifiers implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "SimpleRNNClassifier",
    "SimpleRNNClassifierTorch",
]

from sktime.classification.deep_learning.rnn._rnn_tf import SimpleRNNClassifier
from sktime.classification.deep_learning.rnn._rnn_torch import SimpleRNNClassifierTorch
