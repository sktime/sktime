"""Tests for HampelDetector."""

import pandas as pd
import pytest

from sktime.detection.hampel import HampelDetector
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_hampel_detector_series():
    """Test HampelDetector on univariate Series."""
    X = pd.Series([1.0, 1.0, 10.0, 1.0, 1.0])
    detector = HampelDetector(window_length=3)
    res = detector.fit_predict(X)

    assert isinstance(res, pd.DataFrame)
    assert "ilocs" in res.columns
    assert list(res["ilocs"]) == [2]

    scores = detector.transform_scores(X)
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (5, 1)
    assert scores.iloc[2, 0] > 10.0


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_hampel_detector_panel_support():
    """Test HampelDetector on panel data with MultiIndex (issue #11615)."""
    index = pd.MultiIndex.from_product([["a", "b"], range(5)])
    combined = pd.DataFrame({"value": [1.0, 1.0, 10.0, 1.0, 1.0] * 2}, index=index)

    detector = HampelDetector(window_length=3)
    res = detector.fit_predict(combined)

    assert isinstance(res, pd.DataFrame)
    assert "ilocs" in res.columns
    # Outliers: index 2 in instance 'a' and index 7 (index 2 in instance 'b')
    assert list(res["ilocs"]) == [2, 7]

    scores = detector.transform_scores(combined)
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (10, 1)
    assert (scores.index == combined.index).all()
    # Outlier points have very high anomaly scores
    assert scores.iloc[2, 0] > 10.0
    assert scores.iloc[7, 0] > 10.0
    # Normal points have low anomaly scores
    assert scores.iloc[0, 0] < 1.0
    assert scores.iloc[5, 0] < 1.0
