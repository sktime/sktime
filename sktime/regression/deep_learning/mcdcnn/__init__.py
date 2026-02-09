"""MCDCNN deep learning regressors.

This subpackage provides MCDCNN based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MCDCNNRegressor",
    "MCDCNNRegressorTorch",
]

from sktime.regression.deep_learning.mcdcnn._mcdcnn_tf import MCDCNNRegressor
from sktime.regression.deep_learning.mcdcnn._mcdcnn_torch import MCDCNNRegressorTorch
