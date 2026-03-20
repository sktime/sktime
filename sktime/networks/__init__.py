"""Deep learning network architectures.

Used for both classification and regression tasks.
"""

__all__ = [
    "RNNNetwork",
    "RNNNetworkTorch",
    "MLPAutoencoderTorch",
]

from sktime.networks.rnn import (
    RNNNetwork,
    RNNNetworkTorch,
)

from sktime.networks.autoencoder import MLPAutoencoderTorch
