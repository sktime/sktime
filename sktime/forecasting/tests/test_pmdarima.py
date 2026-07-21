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


@pytest.mark.skipif(
    not run_test_for_class(ARIMA),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARIMA_exogenous_probabilistic():
    """Test ARIMA predict_interval and predict_quantiles with exogenous variables.

    Failure condition of #7849.
    """
    import numpy as np
    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    forecaster = ARIMA()
    forecaster.fit(y_train, X=X_train)
    # X_test has length 4, but fh has length 2.
    # predict_interval and predict_quantiles should slice X_test to match fh size
    fh = [1, 2]
    forecaster.predict_interval(fh=fh, X=X_test, coverage=[0.90])
    forecaster.predict_quantiles(fh=fh, X=X_test, alpha=[0.05, 0.95])


