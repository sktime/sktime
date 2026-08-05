"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "CNNNetwork",
    "CNNNetworkTorch",
    "ResNetNetwork",
    "ResNetNetworkTorch",
    "RNNNetwork",
    "RNNNetworkTorch",
]

from sktime.networks.cnn import (
    CNNNetwork,
    CNNNetworkTorch,
)
from sktime.networks.resnet import (
    ResNetNetwork,
    ResNetNetworkTorch,
)
from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)
