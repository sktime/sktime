"""Tests for SBAForecaster estimator."""

import numpy as np

from sktime.forecasting.sba import SBAForecaster


def test_sba_forecast_less_than_croston():
    """SBA should forecast lower than Croston (bias correction)."""
    from sktime.forecasting.croston import Croston

    y = np.array([0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 4], dtype=float)

    croston = Croston(smoothing=0.2)
    sba = SBAForecaster(smoothing=0.2)

    croston.fit(y, fh=[1])
    sba.fit(y, fh=[1])

    pred_croston = croston.predict()
    pred_sba = sba.predict()

    assert (
        pred_sba[0, 0] < pred_croston[0, 0]
    ), "SBA should produce a lower (bias-corrected) forecast than Croston"


def test_sba_correction_factor():
    """Verify SBA forecast = 0.9 * Croston forecast when smoothing=0.2."""
    from sktime.forecasting.croston import Croston

    y = np.array([0, 3, 0, 0, 5, 0, 2, 0], dtype=float)
    alpha = 0.2

    croston = Croston(smoothing=alpha).fit(y, fh=[1])
    sba = SBAForecaster(smoothing=alpha).fit(y, fh=[1])

    ratio = sba.predict()[0, 0] / croston.predict()[0, 0]
    expected = 1 - alpha / 2

    assert abs(ratio - expected) < 1e-10


def test_sba_get_test_params():
    """Test that get_test_params returns valid parameter dicts."""
    params = SBAForecaster.get_test_params()
    assert isinstance(params, list)
    assert all("smoothing" in p or p == {} for p in params)
