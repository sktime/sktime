"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "RNNNetwork",
    "RNNNetworkTorch",
    "CNTCNetwork",
    "CNTCNetworkTorch",
]

from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)

from sktime.networks.cntc import (
    CNTCNetwork,
    CNTCNetworkTorch,
)