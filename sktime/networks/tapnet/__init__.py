"""TapNet deep learning network implemented in TensorFlow & PyTorch backends."""

__all__ = [
    "TapNetNetwork",
    "TapNetNetworkTorch",
]

from sktime.networks.tapnet._tapnet_tf import TapNetNetwork
from sktime.networks.tapnet._tapnet_torch import TapNetNetworkTorch
