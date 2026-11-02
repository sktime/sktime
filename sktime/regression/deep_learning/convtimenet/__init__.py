"""ConvTimeNet deep learning regressors.

This subpackage provides ConvTimeNet regressors implemented in
PyTorch.
"""

__all__ = [
    "ConvTimeNetRegressor",
]

from sktime.regression.deep_learning.convtimenet._convtimenet_torch import (
    ConvTimeNetRegressor,
)
