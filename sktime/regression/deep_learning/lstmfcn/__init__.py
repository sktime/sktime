"""LSTM-FCN deep learning regressors.

This subpackage provides LSTM-FCN based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "LSTMFCNRegressor",
    "LSTMFCNRegressorTorch",
]

from sktime.regression.deep_learning.lstmfcn._lstmfcn_tf import LSTMFCNRegressor
from sktime.regression.deep_learning.lstmfcn._lstmfcn_torch import (
    LSTMFCNRegressorTorch,
)
