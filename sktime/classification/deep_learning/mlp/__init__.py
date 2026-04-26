"""MLP (Multi-Layer Perceptron) classifier for time series classification in PyTorch."""

__all__ = [
    "MLPClassifier",
    "MLPClassifierTorch",
]

from sktime.classification.deep_learning.mlp._mlp_torch import MLPClassifierTorch


class MLPClassifier(MLPClassifierTorch):
    """PyTorch implementation of ``MLPClassifier``.

    This class keeps the legacy estimator name while using the
    ``MLPClassifierTorch`` backend.
    """
