"""Naive detectors for simple baselines and use in composites or pipelines."""

from sktime.detection.naive._threshold import ThresholdDetector
from sktime.detection.naive._pretrain_window import NaivePretrainWindowDetector

__all__ = [
    "ThresholdDetector",
    "NaivePretrainWindowDetector",
]
