"""Compositions for clusterers."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["ClustererPipeline", "SklearnClustererPipeline"]

from sktime.clustering.compose._pipeline import (
    ClustererPipeline,
    SklearnClustererPipeline,
)
