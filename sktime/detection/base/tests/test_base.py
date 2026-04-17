"""Tests for base change point detection class."""

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_fit_transform_numpy():
    """Test that numpy input works for fit_transform.

    Failure case of bug #8325.
    """
    from sktime.detection.lof import SubLOF

    data = np.array([0, 0.5, 2, 0.1, 0, 0, 0, 2, 0, 0, 0.3, -1, 0, 2, 0.2])
    model = SubLOF(3, window_size=5, novelty=True)
    pred = model.fit_transform(data)

    assert pred.shape[0] == data.shape[0]

    # also test fit alone
    model = SubLOF(3, window_size=5, novelty=True)
    model.fit(data)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_sparse_to_dense_segmentation_series_input():
    """Test sparse_to_dense correctly handles Series with IntervalIndex input.

    Regression test for bugs fixed in GH issue #9917, where sparse_to_dense
    silently misrouted a pd.Series with IntervalIndex to the anomaly/changepoint
    branch, returning all zeros instead of segment labels (Bug 1), and
    _sparse_segments_to_dense returned a DataFrame instead of a Series (Bug 2).
    """
    from sktime.detection.base import BaseDetector

    y_sparse = pd.Series(
        [1, 2, 1],
        index=pd.IntervalIndex.from_arrays([0, 4, 6], [4, 6, 10], closed="left"),
    )
    result = BaseDetector.sparse_to_dense(y_sparse, index=range(10))

    # Bug 1: should return segment labels, not all zeros
    expected = pd.Series([1, 1, 1, 1, 2, 2, 1, 1, 1, 1], dtype="int64")
    pd.testing.assert_series_equal(result, expected)

    # Bug 2: return type must be Series, not DataFrame
    assert isinstance(result, pd.Series)
