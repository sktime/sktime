"""Tests for BaseDetector input type handling."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.base import BaseDetector


class _DummyDetectorWithAsserts(BaseDetector):
    """Dummy detector that asserts internal methods receive pd.DataFrame."""

    _tags = {
        "task": "change_point_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
    }

    def _fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), f"_fit expects pd.DataFrame, got {type(X)}"
        return self

    def _predict(self, X):
        assert isinstance(
            X, pd.DataFrame
        ), f"_predict expects pd.DataFrame, got {type(X)}"
        return pd.Series([5, 10])

    def _predict_points(self, X):
        assert isinstance(
            X, pd.DataFrame
        ), f"_predict_points expects pd.DataFrame, got {type(X)}"
        return pd.Series([5, 10])

    def _predict_segments(self, X):
        assert isinstance(
            X, pd.DataFrame
        ), f"_predict_segments expects pd.DataFrame, got {type(X)}"
        return pd.Series([0, 1])


class _DummySegmentationDetector(BaseDetector):
    """Dummy segmentation detector."""

    _tags = {
        "task": "segmentation",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
    }

    def _fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), f"_fit expects pd.DataFrame, got {type(X)}"
        return self

    def _predict(self, X):
        assert isinstance(
            X, pd.DataFrame
        ), f"_predict expects pd.DataFrame, got {type(X)}"
        return pd.Series([0, 1])

    def _predict_segments(self, X):
        assert isinstance(
            X, pd.DataFrame
        ), f"_predict_segments expects pd.DataFrame, got {type(X)}"
        return pd.Series([0, 1])


def test_predict_points_with_numpy_array():
    """Test predict_points with np.ndarray input."""
    detector = _DummyDetectorWithAsserts()
    X_df = pd.DataFrame({"a": np.arange(20)})
    detector.fit(X_df)

    X_np = np.arange(20).reshape(-1, 1)
    result = detector.predict_points(X_np)

    assert isinstance(result, pd.DataFrame)


def test_predict_segments_with_numpy_array():
    """Test predict_segments with np.ndarray input for change point detection."""
    detector = _DummyDetectorWithAsserts()
    X_df = pd.DataFrame({"a": np.arange(20)})
    detector.fit(X_df)

    X_np = np.arange(20).reshape(-1, 1)
    result = detector.predict_segments(X_np)

    assert isinstance(result, pd.DataFrame)


def test_predict_segments_segmentation_with_numpy_array():
    """Test predict_segments with np.ndarray input for segmentation task."""
    detector = _DummySegmentationDetector()
    X_df = pd.DataFrame({"a": np.arange(20)})
    detector.fit(X_df)

    X_np = np.arange(20).reshape(-1, 1)
    result = detector.predict_segments(X_np)

    assert isinstance(result, pd.DataFrame)


def test_predict_with_numpy_array():
    """Test predict with np.ndarray input."""
    detector = _DummyDetectorWithAsserts()
    X_df = pd.DataFrame({"a": np.arange(20)})
    detector.fit(X_df)

    X_np = np.arange(20).reshape(-1, 1)
    result = detector.predict(X_np)

    assert isinstance(result, pd.DataFrame)

