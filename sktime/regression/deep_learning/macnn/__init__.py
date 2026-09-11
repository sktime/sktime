"""MACNN deep learning regressors.

This subpackage provides MACNN based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MACNNRegressor",
    "MACNNRegressorTorch",
]

from sktime.regression.deep_learning.macnn._macnn_tf import MACNNRegressor
from sktime.regression.deep_learning.macnn._macnn_torch import MACNNRegressorTorch
