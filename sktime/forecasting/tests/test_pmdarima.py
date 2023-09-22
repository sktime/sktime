"""Tests for the ARIMA estimator and _PmdArimaAdapter."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARIMA_pred_quantiles_insample():
    """Test ARIMA predict_quantiles with in-sample fh.

    Failure condition of #4468.
    """
    y = load_airline()
    forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))
    forecaster.fit(y)
    forecaster.predict_quantiles(fh=y.index, X=None, alpha=[0.05, 0.95])
