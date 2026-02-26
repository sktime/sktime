"""Abstract base class for deep learning neural network regressors."""

__all__ = [
    "BaseDeepRegressor",
    "BaseDeepRegressorTorch",
]

from sktime.regression.deep_learning.base._base_tf import BaseDeepRegressor
from sktime.regression.deep_learning.base._base_torch import (
    BaseDeepRegressorTorch,
)
