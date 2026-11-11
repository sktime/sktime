"""Tests for base change point detection class."""

import numpy as np
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
    from sktime.detection.dummy import DummyRegularAnomalies

    data = np.array([0, 0.5, 2, 0.1, 0, 0, 0, 2, 0, 0, 0.3, -1, 0, 2, 0.2])
    model = DummyRegularAnomalies()
    pred = model.fit_transform(data)

    assert pred.shape[0] == data.shape[0]

    # also test fit alone
    model = DummyRegularAnomalies()
    model.fit(data)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_series():
    """Test that pd.Series input works for update and update_predict (#11606)."""
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    s1 = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    s2 = pd.Series([4.0, 5.0], index=pd.date_range("2020-01-04", periods=2))

    model = DummyRegularAnomalies()
    model.fit(s1)
    res = model.update(s2)
    assert res is model
    assert len(model._X) == 5

    pred = model.update_predict(s2)
    assert isinstance(pred, pd.DataFrame)
    assert "ilocs" in pred.columns


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_numpy():
    """Test that np.ndarray input works for update and update_predict (#11606)."""
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    a1 = np.array([1.0, 2.0, 3.0])
    a2 = np.array([4.0, 5.0])

    model = DummyRegularAnomalies()
    model.fit(a1)
    res = model.update(a2)
    assert res is model
    assert len(model._X) == 5
    np.testing.assert_allclose(model._X.values.flatten(), [1.0, 2.0, 3.0, 4.0, 5.0])

    pred = model.update_predict(a2)
    assert isinstance(pred, pd.DataFrame)
    assert "ilocs" in pred.columns


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_update_with_y():
    """Test that update with y works for pd.Series and np.ndarray (#11606)."""
    import pandas as pd

    from sktime.detection.dummy import DummyRegularAnomalies

    s1 = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    s2 = pd.Series([4.0, 5.0], index=pd.date_range("2020-01-04", periods=2))
    y1 = pd.Series([0, 0, 1], index=pd.date_range("2020-01-01", periods=3))
    y2 = pd.Series([0, 1], index=pd.date_range("2020-01-04", periods=2))

    model = DummyRegularAnomalies()
    model.fit(s1, y1)
    model.update(s2, y2)
    assert len(model._y) == 5
