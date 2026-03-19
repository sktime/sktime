"""FCN networks for time series classification and regression."""

__all__ = [
    "FCNNetwork",
    "FCNNetworkTorch",
]

from sktime.networks.fcn._fcn_tf import FCNNetwork
from sktime.networks.fcn._fcn_torch import FCNNetworkTorch
