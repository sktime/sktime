"""CNN deep learning regressors.

This subpackage provides CNN based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "CNNRegressor",
    "CNNRegressorTorch",
]

from sktime.regression.deep_learning.cnn._cnn_tf import CNNRegressor
from sktime.regression.deep_learning.cnn._cnn_torch import CNNRegressorTorch
