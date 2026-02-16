"""Composition for outlier, changepoint, segmentation estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.detection.compose._as_transform import (
    AnnotatorAsTransformer,
    DetectorAsTransformer,
)
from sktime.detection.compose._pipeline import AnnotatorPipeline, DetectorPipeline

__all__ = [
    "AnnotatorAsTransformer",
    "AnnotatorPipeline",
    "DetectorAsTransformer",
    "DetectorPipeline",
]
