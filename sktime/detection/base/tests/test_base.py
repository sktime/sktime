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
def test_transform_preserves_index():
    """Test that transform preserves input index.

    Regression test for issue: BaseDetector.transform drops original time index.
    """
    from sktime.detection.naive import ThresholdDetector

    rng = pd.date_range("2024-01-01", periods=5, freq="D")
    X = pd.Series([1, 2, 10, 2, 1], index=rng)

    model = ThresholdDetector(upper=5)
    model.fit(X)
    y_pred = model.transform(X)

    assert all(y_pred.index == X.index)
