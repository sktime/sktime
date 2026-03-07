"""FCN for time series classification and regression in TensorFlow and PyTorch backends."""

__all__ = [
        "FCNNetwork",
        "FCNNetworkTorch",
]

from sktime.networks.fcn._fcn_tf import FCNNetwork
from sktime.networks.fcn._fcn_torch import FCNNetworkTorch