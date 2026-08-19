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


@pytest.mark.skipif(
    not run_test_for_class(TSB),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsb_get_fitted_params():
    """Test that the recursion state is exposed via get_fitted_params."""
    y = load_PBS_dataset()
    forecaster = TSB(alpha=0.4, beta=0.05).fit(y)

    fitted_params = forecaster.get_fitted_params()

    expected = {"demand_size", "demand_probability", "forecast"}
    assert expected <= set(fitted_params)
    assert fitted_params["demand_size"] == forecaster._d_last
    assert fitted_params["demand_probability"] == forecaster._p_last
    assert fitted_params["forecast"] == forecaster._f[-1]
    # forecast is the product of the two, by construction of the method
    np.testing.assert_allclose(
        fitted_params["forecast"],
        fitted_params["demand_size"] * fitted_params["demand_probability"],
        rtol=1e-12,
    )
