"""Tests for FLUSSSegmenter."""

import pytest

from sktime.detection._fluss import FLUSSSegmenter


def test_fluss_predict_output_shape():
    """FLUSSSegmenter should return sparse format with ``ilocs`` column."""
    pytest.importorskip("stumpy")
    import pandas as pd

    X = pd.DataFrame({"x": list(range(40))})
    est = FLUSSSegmenter(window_length=3, n_regimes=2, excl_factor=1)
    y = est.fit_predict(X)

    assert isinstance(y, pd.DataFrame)
    assert "ilocs" in y.columns


def test_fluss_predict_detects_shift():
    """FLUSSSegmenter should detect a clear regime shift."""
    pytest.importorskip("stumpy")
    import numpy as np
    import pandas as pd

    data = np.concatenate([np.ones(30) * 40, np.ones(30) * 10])
    X = pd.DataFrame({"x": data})

    est = FLUSSSegmenter(window_length=5, n_regimes=2, excl_factor=1)
    y = est.fit_predict(X)

    assert "ilocs" in y.columns
    change_points = y["ilocs"].tolist()

    assert len(change_points) > 0
    assert 25 <= change_points[0] <= 35
