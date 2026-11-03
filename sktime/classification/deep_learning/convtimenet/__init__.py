"""ConvTimeNet deep learning classifiers.

This subpackage provides ConvTimeNet classifiers implemented in
PyTorch.
"""

__all__ = [
    "ConvTimeNetClassifier",
]

from sktime.classification.deep_learning.convtimenet._convtimenet_torch import (
    ConvTimeNetClassifier,
)
