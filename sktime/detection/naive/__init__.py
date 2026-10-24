"""Naive detectors for simple baselines and use in composites or pipelines."""

from sktime.detection.naive._quantile import QuantileDetector
from sktime.detection.naive._threshold import ThresholdDetector

__all__ = [
    "QuantileDetector",
    "ThresholdDetector",
]
