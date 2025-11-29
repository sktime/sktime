"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "FCNNetwork",
    "FCNNetworkTorch",
    "RNNNetwork",
    "RNNNetworkTorch",
]

from sktime.networks.fcn import (
    FCNNetwork,
    FCNNetworkTorch,
)
from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)
