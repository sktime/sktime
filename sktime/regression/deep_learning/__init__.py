# -*- coding: utf-8 -*-
"""Deep learning based regressors."""
__all__ = [
    "CNNRegressor",
    "TapNetRegressor",
]

from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.regression.deep_learning.tapnet import TapNetRegressor
