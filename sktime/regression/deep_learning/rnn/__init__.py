"""RNN deep learning regressors.

This subpackage provides RNN based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "SimpleRNNRegressor",
    "SimpleRNNRegressorTorch",
]

from sktime.regression.deep_learning.rnn._rnn_tf import SimpleRNNRegressor
from sktime.regression.deep_learning.rnn._rnn_torch import SimpleRNNRegressorTorch
