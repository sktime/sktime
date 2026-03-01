"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "ResNetNetwork",
    "ResNetNetworkTorch",
    "RNNNetwork",
    "RNNNetworkTorch",
]

from sktime.networks.resnet import (
    ResNetNetwork,
    ResNetNetworkTorch,
)
from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)
