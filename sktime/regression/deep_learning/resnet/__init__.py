"""ResNet deep learning regressors.

This subpackage provides ResNet based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "ResNetRegressor",
    "ResNetRegressorTorch",
]

from sktime.regression.deep_learning.resnet._resnet_tf import ResNetRegressor
from sktime.regression.deep_learning.resnet._resnet_torch import ResNetRegressorTorch
