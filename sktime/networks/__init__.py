"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "CNNNetwork",
    "CNNNetworkTorch",
    "RNNNetwork",
    "RNNNetworkTorch",
    "MLPAutoencoderTorch",
]

from sktime.networks.cnn import (
    CNNNetwork,
    CNNNetworkTorch,
)
from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)

from sktime.networks.autoencoder import MLPAutoencoderTorch
