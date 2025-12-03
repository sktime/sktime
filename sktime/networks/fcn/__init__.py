"""FCN deep learning network structure implemented in TensorFlow & PyTorch backends."""

__all__ = [
    "FCNNetwork",
    "FCNNetworkTorch",
]

from sktime.networks.fcn._fcn_tf import FCNNetwork
from sktime.networks.fcn._fcn_torch import FCNNetworkTorch
