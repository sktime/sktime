"""Composition for outlier, changepoint, segmentation estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.annotation.compose._as_transform import AnnotatorAsTransformer
from sktime.annotation.compose._pipeline import AnnotatorPipeline

__all__ = ["AnnotatorAsTransformer", "AnnotatorPipeline"]
