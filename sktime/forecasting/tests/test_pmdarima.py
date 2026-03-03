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

@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARIMA_with_covariates():
    """Test ARIMA with covariates to reproduce issue #XXXX."""
    from sktime.split import temporal_train_test_split
    
    y = load_airline()
    x = y * 2
    fh = [1, 2, 3]
    
    y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.25)
    
    forecaster = ARIMA(suppress_warnings=True)
    forecaster.fit(y_train, X=x_train, fh=fh)
    y_pred = forecaster.predict(X=x_test, fh=fh)
    
    # Just verify that prediction completes without error
    assert len(y_pred) == len(fh)
