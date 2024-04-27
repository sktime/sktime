"""Tests for the BaseSeriesAnnotator class."""

__author__ = ["Alex-JG3"]
__all__ = []

import pandas as pd
import pytest

from sktime.annotation.base._base import BaseSeriesAnnotator


@pytest.mark.parametrize(
    "y_sparse, y_dense_expected, length",
    [
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1], dtype="int32"), None),
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1, 0, 0], dtype="int32"), 6),
        (
            pd.DataFrame({"seg_label": [1, 2], "seg_start": [0, 3], "seg_end": [2, 4]}),
            pd.Series([1, 1, 1, 2, 2], dtype="int32"),
            None,
        ),
    ],
)
def test_sparse_to_dense(y_sparse, y_dense_expected, length):
    """Test converting from sparse to dense."""
    y_dense_actual = BaseSeriesAnnotator.sparse_to_dense(y_sparse, length=length)
    assert y_dense_actual.equals(y_dense_expected)
