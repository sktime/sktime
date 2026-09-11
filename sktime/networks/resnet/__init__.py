"""ResNet deep learning network structure.

Implemented in TensorFlow & PyTorch backends.
"""

__all__ = [
    "ResNetNetwork",
    "ResNetNetworkTorch",
]

from sktime.networks.resnet._resnet_tf import ResNetNetwork
from sktime.networks.resnet._resnet_torch import ResNetNetworkTorch
