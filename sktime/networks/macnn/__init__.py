"""MACNN deep learning network structures.

This subpackage provides MACNN network implementations in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "MACNNNetwork",
    "MACNNNetworkTorch",
]

from sktime.networks.macnn._macnn_tf import MACNNNetwork
from sktime.networks.macnn._macnn_torch import MACNNNetworkTorch
