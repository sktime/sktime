"""MCDCNN deep learning classifiers implemented in PyTorch."""

__all__ = [
    "MCDCNNClassifier",
    "MCDCNNClassifierTorch",
]

from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import (
    MCDCNNClassifierTorch,
)


class MCDCNNClassifier(MCDCNNClassifierTorch):
    """PyTorch implementation of ``MCDCNNClassifier``.

    This class keeps the legacy estimator name while using the
    ``MCDCNNClassifierTorch`` backend.
    """
