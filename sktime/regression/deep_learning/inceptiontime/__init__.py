"""InceptionTime deep learning regressors.

This subpackage provides InceptionTime based regressors implemented in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "InceptionTimeRegressor",
    "InceptionTimeRegressorTorch",
]

from sktime.regression.deep_learning.inceptiontime._inceptiontime_tf import (
    InceptionTimeRegressor,
)
from sktime.regression.deep_learning.inceptiontime._inceptiontime_torch import (
    InceptionTimeRegressorTorch,
)
