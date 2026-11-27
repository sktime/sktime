"""GRU network structures for time series classification.

Implemented in PyTorch backend.
"""

__all__ = ["GRUNetworkTorch", "GRUFCNNNetworkTorch"]

from sktime.networks.gru._gru_fcnn_torch import GRUFCNNNetworkTorch
from sktime.networks.gru._gru_torch import GRUNetworkTorch
