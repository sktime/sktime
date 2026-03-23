"""CNTC Deep Learning Regressor.

This submodule provides the CNTC deep learning regressor
implemented in both TensorFlow and PyTorch backends.
"""

__all__ = [
    "CNTCRegressor",
    "CNTCRegressorTorch",
]

from sktime.regression.deep_learning.cntc._cntc_tf import CNTCRegressor
from sktime.regression.deep_learning.cntc._cntc_torch import CNTCRegressorTorch
