"""TapNet deep learning regressors.

This subpackage provides TapNet based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "TapNetRegressor",
    "TapNetRegressorTorch",
]

from sktime.regression.deep_learning.tapnet._tapnet_tf import TapNetRegressor
from sktime.regression.deep_learning.tapnet._tapnet_torch import TapNetRegressorTorch
