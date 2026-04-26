"""RNN deep learning classifiers implemented in PyTorch."""

__all__ = [
    "SimpleRNNClassifier",
    "SimpleRNNClassifierTorch",
]

from sktime.classification.deep_learning.rnn._rnn_torch import SimpleRNNClassifierTorch


class SimpleRNNClassifier(SimpleRNNClassifierTorch):
    """PyTorch implementation of ``SimpleRNNClassifier``.

    This class keeps the legacy estimator name while using the
    ``SimpleRNNClassifierTorch`` backend.
    """
