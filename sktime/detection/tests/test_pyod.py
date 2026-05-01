"""Tests for PyOD-based detectors."""

import pandas as pd
from pyod.models.ecod import ECOD

from sktime.detection.adapters._pyod import PyODDetector


def test_pyod_detector_score_returns_float():
    X = pd.DataFrame([[1, 2], [3, 4], [10, 11]])

    model = PyODDetector(ECOD(), labels="score")
    y = model.fit_transform(X)

    assert y.dtypes.iloc[0] == float