"""MCDCNN deep learning network implemented in TensorFlow & PyTorch backends."""

__all__ = [
    "MCDCNNNetwork",
    "MCDCNNNetworkTorch",
]

from sktime.networks.mcdcnn._mcdcnn_tf import MCDCNNNetwork
from sktime.networks.mcdcnn._mcdcnn_torch import MCDCNNNetworkTorch
