"""LSTM-FCN deep learning classifiers implemented in PyTorch."""

__all__ = [
    "LSTMFCNClassifier",
    "LSTMFCNClassifierTorch",
]

from sktime.classification.deep_learning.lstmfcn._lstmfcn_torch import (
    LSTMFCNClassifierTorch,
)


class LSTMFCNClassifier(LSTMFCNClassifierTorch):
    """PyTorch implementation of ``LSTMFCNClassifier``.

    This class keeps the legacy estimator name while using the
    ``LSTMFCNClassifierTorch`` backend.
    """
