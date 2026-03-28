"""Tests for the HampelFilter outlier detector."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.hampel import HampelFilter


def _make_series_with_outlier(outlier_idx=5, outlier_val=100.0, length=15):
    """Create a simple time series with one planted outlier."""
    values = np.ones(length) * 1.5
    values[outlier_idx] = outlier_val
    return pd.Series(values)


def test_hampel_detects_obvious_outlier():
    """HampelFilter should flag a large outlier in an otherwise flat series."""
    X = _make_series_with_outlier(outlier_idx=5, outlier_val=100.0)
    detector = HampelFilter(window_size=3, n_sigma=3.0)
    result = detector.fit_predict(X)

    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns
    assert 5 in result["ilocs"].values, "Obvious outlier at index 5 should be in ilocs"
    assert len(result) == 1, "Only one outlier should be detected"


def test_hampel_no_false_positives_constant_series():
    """HampelFilter should return empty result for a constant series."""
    X = pd.Series(np.ones(20) * 5.0)
    detector = HampelFilter(window_size=5, n_sigma=3.0)
    result = detector.fit_predict(X)
    assert len(result) == 0, "No outliers should be detected in a constant series"


def test_hampel_returns_sparse_dataframe():
    """Output should be a sparse DataFrame with ilocs column."""
    idx = pd.date_range("2024-01-01", periods=15, freq="D")
    values = np.ones(15) * 2.0
    values[7] = 999.0
    X = pd.Series(values, index=idx)
    detector = HampelFilter(window_size=3, n_sigma=3.0)
    result = detector.fit_predict(X)

    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns
    assert len(result) < len(X)
    assert 7 in result["ilocs"].values


def test_hampel_ilocs_are_valid_positions():
    """All ilocs values must be non-negative integers within range of X."""
    X = _make_series_with_outlier(length=20)
    result = HampelFilter().fit_predict(X)
    assert result["ilocs"].dtype in [np.int64, np.int32, "int64", "int32"]
    assert (result["ilocs"] >= 0).all()
    assert (result["ilocs"] < len(X)).all()


def test_hampel_empty_output_structure():
    """Empty output should still be a DataFrame with ilocs column."""
    X = pd.Series(np.ones(10) * 3.0)
    result = HampelFilter(window_size=3, n_sigma=3.0).fit_predict(X)
    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns
    assert len(result) == 0


@pytest.mark.parametrize("window_size", [1, 3, 7])
def test_hampel_various_window_sizes(window_size):
    """HampelFilter should run without error for different window sizes."""
    X = _make_series_with_outlier(length=20)
    detector = HampelFilter(window_size=window_size, n_sigma=3.0)
    result = detector.fit_predict(X)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(X)


def test_hampel_sktime_compliance():
    """HampelFilter must pass sktime's generic estimator interface checks."""
    from sktime.utils.estimator_checks import check_estimator
    check_estimator(HampelFilter(window_size=3), raise_exceptions=True)
