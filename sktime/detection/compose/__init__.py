"""Composition for outlier, changepoint, segmentation estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sktime.detection.compose._as_transform import DetectorAsTransformer
from sktime.detection.compose._pipeline import DetectorPipeline
from sktime.detection.compose._stream_calibrate_fpr import StreamCalibrateFPR

__all__ = [
    "DetectorAsTransformer",
    "DetectorPipeline",
    "StreamCalibrateFPR",
]
