"""CNN deep learning network structure implemented in TensorFlow & PyTorch backends."""

__all__ = [
    "CNNNetwork",
    "CNNNetworkTorch",
]

from sktime.networks.cnn._cnn_tf import CNNNetwork
from sktime.networks.cnn._cnn_torch import CNNNetworkTorch
