"""InceptionTime deep learning network structures.

This subpackage provides InceptionTime network implementations in
TensorFlow and PyTorch backends.
"""

__all__ = [
    "InceptionTimeNetwork",
    "InceptionTimeNetworkTorch",
]

from sktime.networks.inceptiontime._inceptiontime_tf import InceptionTimeNetwork
from sktime.networks.inceptiontime._inceptiontime_torch import (
    InceptionTimeNetworkTorch,
)
