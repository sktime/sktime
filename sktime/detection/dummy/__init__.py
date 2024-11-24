"""Dummy detectors for baselines and testing."""

from sktime.detection.dummy._dummy_regular_cp import DummyRegularChangePoints
from sktime.detection.dummy._zero_cp import ZeroChangePoints

__all__ = [
    "DummyRegularChangePoints",
    "ZeroChangePoints",
]
