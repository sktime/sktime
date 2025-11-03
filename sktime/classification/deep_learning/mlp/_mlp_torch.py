"""MLP (Multi-Layer Perceptron) Classifier for Time Series Classification in PyTorch."""

__authors__ = ["RecreationalMath"]
__all__ = ["MLPClassifierTorch"]

from collections.abc import Callable

from sktime.classification.deep_learning.base import BaseDeepClassifierTorch
from sktime.networks.mlp import MLPNetwork