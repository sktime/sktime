"""Tests for TSB estimator."""

import numpy as np
import pytest

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.tsb import TSB
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "alpha, beta, fh, expected_forecast",
    [
        (0.1, 0.1, np.array([10]), 0.015194),
        (0.4, 0.05, np.array([5]), 0.106244),
        (0.5, 0.05, np.array([15]), 0.116315),
    ],
)
def test_TSB(alpha, beta, fh, expected_forecast):
    """
    Test TSB forecaster.
    """
    y = load_PBS_dataset()
    forecaster = TSB(alpha, beta)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=fh)
    np.testing.assert_almost_equal(
        y_pred, np.full(len(fh), expected_forecast), decimal=5
    )
