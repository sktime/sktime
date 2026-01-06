"""MLP deep learning regressors.

This subpackage provides MLP based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MLPRegressor",
    "MLPRegressorTorch",
]

from sktime.regression.deep_learning.mlp._mlp_tf import MLPRegressor
from sktime.regression.deep_learning.mlp._mlp_torch import MLPRegressorTorch
