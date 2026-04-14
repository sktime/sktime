"""MLP (Multi-Layer Perceptron) Regressor for Time Series Regression.

This subpackage provides Multi-Layer Perceptron (MLP) based time series
regressor in TensorFlow and PyTorch backends.
"""

__all__ = [
    "MLPRegressor",
    "MLPRegressorTorch",
]
from sktime.regression.deep_learning.mlp._mlp_tf import MLPRegressor
from sktime.regression.deep_learning.mlp._mlp_torch import MLPRegressorTorch
