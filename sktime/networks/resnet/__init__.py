"""ResNet deep learning network structure."""

__all__ = [
    "ResNetNetwork",
    "ResNetNetworkTorch",
]

from sktime.networks.resnet._resnet_tf import ResNetNetwork
from sktime.networks.resnet._resnet_torch import ResNetNetworkTorch
