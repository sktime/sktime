"""MLP (Multi-Layer Perceptron) deep learning network structure.

Implemented in TensorFlow & PyTorch backends.
"""

__all__ = [
    "MLPNetwork",
    "MLPNetworkTorch",
]

from sktime.networks.mlp._mlp_tf import MLPNetwork
from sktime.networks.mlp._mlp_torch import MLPNetworkTorch