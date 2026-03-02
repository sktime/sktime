"""FCN time series regressors.

This subpackage provides FCN based time series regressors implemented in TensorFlow and PyTorch backends.  
"""

__all__ = [
    "FCNRegressor",
    "FCNRegressorTorch",
]   

from sktime.regression.deep_learning.fcn._fcn_tf import FCNRegressor
from sktime.regression.deep_learning.fcn._fcn_torch import FCNRegressorTorch