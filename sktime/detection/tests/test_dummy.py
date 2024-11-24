"""Tests for dummy detectors."""

import pandas as pd


def test_dummy_changepoints():
    """Test expected output for DummyChangePoints."""
    from sktime.detection.dummy._dummy_regular_cp import DummyRegularChangePoints

    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d = DummyRegularChangePoints(step_size=3)
    y_cp_ds = d.fit_transform(y)
    y_cp_sp = d.fit_predict(y)

    expected_ds = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    assert y_cp_ds.equals(expected_ds)

    expected_sp = pd.Series([2, 5, 8])
    assert y_cp_sp.equals(expected_sp)
