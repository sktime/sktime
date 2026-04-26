"""Tests for SubIsolationForest anomaly detector."""

__author__ = ["rupeshca007"]

import datetime

import pandas as pd
import pytest

from sktime.detection.isof import SubIsolationForest
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "x,interval_size,expected_intervals",
    [
        (
            pd.Index([1, 2, 3, 4, 5]),
            1,
            pd.IntervalIndex.from_breaks([1, 2, 3, 4, 5, 6], closed="left"),
        ),
        (
            pd.Index([0.0, 1.3, 1.5, 2.0, 3.5]),
            1.5,
            pd.IntervalIndex.from_breaks([0.0, 1.5, 3.0, 4.5], closed="left"),
        ),
        (
            pd.DatetimeIndex(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                    "2024-01-05 00:00:00",
                ]
            ),
            datetime.timedelta(days=1),
            pd.IntervalIndex.from_breaks(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                    "2024-01-05 00:00:00",
                    "2024-01-06 00:00:00",
                ],
                closed="left",
                dtype="interval[datetime64[ns], left]",
            ),
        ),
    ],
)
def test_split_into_intervals(x, interval_size, expected_intervals):
    """Test that _split_into_intervals produces correct interval partitions."""
    actual_intervals = SubIsolationForest._split_into_intervals(x, interval_size)
    assert (actual_intervals == expected_intervals).all()


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X,expected_ilocs",
    [
        (
            # Points 2, 7, 13 are extreme outliers within their respective windows
            pd.DataFrame([0, 0, 100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 0]),
            [2, 7, 13],
        ),
    ],
)
def test_predict_detects_outliers(X, expected_ilocs):
    """Test that SubIsolationForest detects clear point anomalies."""
    model = SubIsolationForest(
        window_size=5,
        n_estimators=10,
        contamination=0.2,
        random_state=42,
    )
    model.fit(X)
    y_actual = model.predict(X)
    detected = set(y_actual["ilocs"].tolist())
    for iloc in expected_ilocs:
        assert iloc in detected, (
            f"Expected anomaly at iloc={iloc} not found in detected={detected}"
        )


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predict_returns_dataframe():
    """Test that predict() returns a pd.DataFrame with an 'ilocs' column."""
    X = pd.DataFrame([0, 0.5, 100, 0.1, 0, 0.2, 0.3, 50, 0, 0])
    model = SubIsolationForest(window_size=3, n_estimators=5, random_state=0)
    model.fit(X)
    y = model.predict(X)
    assert isinstance(y, pd.DataFrame), "predict() should return a pd.DataFrame"
    assert "ilocs" in y.columns, "predict() result must have an 'ilocs' column"


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_returns_dense_dataframe():
    """Test that fit_transform() returns a dense DataFrame with a 'labels' column."""
    X = pd.DataFrame([0, 0, 100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 0])
    model = SubIsolationForest(window_size=5, n_estimators=10, random_state=42)
    result = model.fit_transform(X)
    assert isinstance(result, pd.DataFrame)
    assert "labels" in result.columns
    assert len(result) == len(X), "fit_transform() output must have same length as X"


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_does_not_mutate_input():
    """Test that SubIsolationForest does not modify the input DataFrame."""
    X = pd.DataFrame([0, 0.5, 100, 0.1, 0, 0.2, 0.3, 50])
    X_original = X.copy(deep=True)

    model = SubIsolationForest(window_size=2, n_estimators=5, random_state=0)
    _ = model.fit_transform(X)

    pd.testing.assert_frame_equal(X, X_original)


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_get_test_params():
    """Test that get_test_params returns valid parameter dicts."""
    params = SubIsolationForest.get_test_params()
    assert isinstance(params, list)
    assert len(params) >= 1
    for p in params:
        assert isinstance(p, dict)
        # Should not raise
        inst = SubIsolationForest(**p)
        assert inst is not None


@pytest.mark.skipif(
    not run_test_for_class(SubIsolationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate():
    """Test that SubIsolationForest handles multivariate time series."""
    X = pd.DataFrame(
        {
            "a": [0, 0, 100, 0, 0, 0, 0, 100, 0, 0],
            "b": [0, 0, 200, 0, 0, 0, 0, 200, 0, 0],
        }
    )
    model = SubIsolationForest(window_size=5, n_estimators=10, random_state=42)
    model.fit(X)
    y = model.predict(X)
    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns
