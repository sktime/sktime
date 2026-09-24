# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for _sparse_points_to_dense with non-default indices.

Covers the bug reported in https://github.com/sktime/sktime/issues/XXXX:
_sparse_points_to_dense crashed with KeyError when the index is a
DatetimeIndex or any non-integer index, because it used label-based []
indexing with iloc integer positions.
"""

__author__ = ["rupeshca007"]

import numpy as np
import pandas as pd
import pytest

from sktime.detection.base import BaseDetector
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sparse_points_to_dense_datetime_index():
    """_sparse_points_to_dense must not raise KeyError with DatetimeIndex.

    Before the fix, the following raised:
        KeyError: 0
    because integer iloc positions were used as datetime label lookups.

    This scenario is realistic for sensor time series (e.g., vibration data
    sampled at millisecond frequency with a DatetimeIndex).
    """
    index = pd.date_range("2024-01-01", periods=10, freq="ms")
    y_sparse = pd.DataFrame({"ilocs": [0, 3, 7]})  # anomalies at positions 0, 3, 7

    # Should NOT raise KeyError
    result = BaseDetector._sparse_points_to_dense(y_sparse, index)

    assert isinstance(result, pd.Series)
    assert len(result) == 10
    assert result.index.equals(index), "Output index must match input DatetimeIndex"

    # Anomaly positions must be labelled 1
    assert result.iloc[0] == 1
    assert result.iloc[3] == 1
    assert result.iloc[7] == 1

    # Non-anomaly positions must be 0
    assert result.iloc[1] == 0
    assert result.iloc[5] == 0
    assert result.iloc[9] == 0


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sparse_points_to_dense_string_index():
    """_sparse_points_to_dense must work with arbitrary string indices."""
    index = pd.Index(["a", "b", "c", "d", "e"])
    y_sparse = pd.DataFrame({"ilocs": [1, 4]})

    result = BaseDetector._sparse_points_to_dense(y_sparse, index)

    assert result.index.tolist() == ["a", "b", "c", "d", "e"]
    assert result.iloc[1] == 1
    assert result.iloc[4] == 1
    assert result.iloc[0] == 0
    assert result.iloc[2] == 0


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sparse_points_to_dense_range_index_unchanged():
    """Existing behaviour for RangeIndex must be preserved (regression guard)."""
    index = pd.RangeIndex(7)
    y_sparse = pd.Series([2, 4])

    result = BaseDetector._sparse_points_to_dense(y_sparse, index)

    expected = pd.Series([0, 0, 1, 0, 1, 0, 0], dtype="int64")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_sparse_points_to_dense_empty_sparse():
    """Empty sparse input must produce all-zeros dense output with correct index."""
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    y_sparse = pd.DataFrame({"ilocs": pd.Series([], dtype=int)})

    result = BaseDetector._sparse_points_to_dense(y_sparse, index)

    assert len(result) == 5
    assert (result == 0).all()
    assert result.index.equals(index)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_transform_preserves_datetime_index():
    """End-to-end: fit_transform output must have same DatetimeIndex as input X.

    This is an integration test confirming the fix works through the full
    fit_transform → sparse_to_dense → _sparse_points_to_dense pipeline.
    """
    from sktime.detection.lof import SubLOF

    index = pd.date_range("2024-01-01", periods=15, freq="D")
    X = pd.DataFrame(
        {"value": [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0]},
        index=index,
    )

    detector = SubLOF(n_neighbors=3, window_size=5, novelty=True)
    result = detector.fit_transform(X)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X)
    assert result.index.equals(index), (
        "fit_transform output must have the same DatetimeIndex as input X"
    )
