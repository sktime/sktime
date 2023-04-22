# -*- coding: utf-8 -*-
"""Tests for the ARIMA estimator and _PmdArimaAdapter."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(ARIMA, severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_ARIMA_pred_quantiles_insample():
    """Test ARIMA predict_quantiles with in-sample fh. Failure condition of #4468."""
    y = load_airline()
    forecaster = ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))
    forecaster.fit(y)
    forecaster.predict_quantiles(fh=y.index, X=None, alpha=[0.05, 0.95])
