"""Multivariate time series transformer deep learning classifiers.

This subpackage provides multivariate time series transformer classifiers
implemented in PyTorch.
"""

__all__ = [
    "MVTSTransformerClassifier",
]

from sktime.classification.deep_learning.mvts_transformer._mvts_transformer_torch import (  # noqa: E501
    MVTSTransformerClassifier,
)
