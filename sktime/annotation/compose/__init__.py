"""Composition for outlier, changepoint, segmentation estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.detection.compose import AnnotatorAsTransformer, AnnotatorPipeline

__all__ = [
    "AnnotatorAsTransformer",
    "AnnotatorPipeline",
]
