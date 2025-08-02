"""Dummy detectors for baselines and testing."""

from sktime.detection.dummy._dummy_regular_an import DummyRegularAnomalies
from sktime.detection.dummy._dummy_regular_cp import DummyRegularChangePoints
from sktime.detection.dummy._zero_an import ZeroAnomalies
from sktime.detection.dummy._zero_cp import ZeroChangePoints
from sktime.detection.dummy._zero_seg import ZeroSegments

__all__ = [
    "DummyRegularAnomalies",
    "DummyRegularChangePoints",
    "ZeroAnomalies",
    "ZeroChangePoints",
    "ZeroSegments",
]
