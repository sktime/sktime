"""CNTC deep learning network structure implemented in TensorFlow and PyTorch backends."""


__all__ = [
    "CNTCNetwork",
    "CNTCNetworkTorch",
]

from sktime.networks.cntc._cntc_tf import CNTCNetwork
from sktime.networks.cntc._cntc_torch import CNTCNetworkTorch
