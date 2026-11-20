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
def test_ARIMA_integer_index_not_starting_at_zero():
    """Test ARIMA out-of-sample prediction with an integer index not starting at 0.

    statsmodels >= 0.15 raises "No supported index is available" for such indices,
    failure condition of #11228.
    """
    import numpy as np
    import pandas as pd

    index = pd.Index(np.arange(3, 43), dtype="int64")
    rng = np.random.default_rng(42)
    y = pd.Series(rng.normal(size=40).cumsum(), index=index)
    X = pd.DataFrame({"x": rng.normal(size=45)}, index=np.arange(3, 48))

    forecaster = ARIMA(order=(1, 0, 0))
    forecaster.fit(y.iloc[:35], X=X.iloc[:35], fh=[1, 2])

    y_pred = forecaster.predict(X=X.loc[38:39])
    assert list(y_pred.index) == [38, 39]

    pred_int = forecaster.predict_interval(X=X.loc[38:39])
    assert list(pred_int.index) == [38, 39]

    forecaster.update(y.iloc[35:], X=X.loc[38:42])
    y_pred = forecaster.predict(X=X.loc[43:44])
    assert list(y_pred.index) == [43, 44]
