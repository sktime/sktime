"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "RNNNetwork",
    "RNNNetworkTorch",
    "FCNNetwork",
    "FCNNetworkTorch",
]

from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)

from sktime.networks.fcn import (
    FCNNetwork,
    FCNNetworkTorch,
)
