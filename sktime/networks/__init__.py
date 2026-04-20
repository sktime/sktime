"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "RNNNetwork",
    "RNNNetworkTorch",
    "TapNetNetwork",
    "TapNetNetworkTorch",
]

from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)
from sktime.networks.tapnet import (
    TapNetNetwork,
    TapNetNetworkTorch,
)
