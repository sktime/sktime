"""Tests the VAR model."""

__author__ = ["thayeylolu", "dhairya-motta"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon

#
from sktime.forecasting.var import VAR
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(VAR),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VAR_against_statsmodels():
    """Compares Sktime's and Statsmodel's VAR."""
    from statsmodels.tsa.api import VAR as _VAR

    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        freq = "ME"
    else:
        freq = "M"

    index = pd.date_range(start="2005", end="2006-12", freq=freq)
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(23, 2)),
        columns=list("AB"),
        index=pd.PeriodIndex(index),
    )
    train, test = temporal_train_test_split(df)
    sktime_model = VAR()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VAR(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.forecast(train.values[-lagged:], steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    assert_allclose(y_pred, new_arr)


@pytest.mark.skipif(
    not run_test_for_class(VAR),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VAR_ic_aic_k_ar_zero():
    """Regression test for #4055: VAR with ic='aic' when k_ar=0 is selected.

    Uses a fixed-seed i.i.d. white-noise dataset where AIC reliably selects
    k_ar=0 (no autoregressive lags), ensuring the statsmodels IndexError fix
    is exercised deterministically on every test run.
    """
    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    freq = "ME" if pandas2 else "M"

    rng = np.random.default_rng(42)
    n_obs = 50

    index = pd.date_range(start="2010", periods=n_obs, freq=freq)
    # Pure i.i.d. white noise: no autocorrelation, so AIC should select k_ar=0
    data = rng.standard_normal((n_obs, 2))
    df = pd.DataFrame(data, columns=["A", "B"], index=pd.PeriodIndex(index))

    forecaster = VAR(ic="aic")
    forecaster.fit(df)

    # Confirm the edge case is actually triggered — if this fails, the dataset
    # no longer forces k_ar=0 and the test needs to be updated
    assert forecaster._fitted_forecaster.k_ar == 0, (
        "Expected k_ar=0 for i.i.d. white noise with ic='aic', "
        f"but got k_ar={forecaster._fitted_forecaster.k_ar}. "
        "The test dataset may need adjustment."
    )

    fh = ForecastingHorizon([1, 3])

    # Both of these must not raise IndexError (the bug this test guards against)
    y_pred = forecaster.predict(fh=fh)
    assert y_pred.shape == (2, 2)

    pred_int = forecaster.predict_interval(fh=fh, coverage=[0.9])
    assert pred_int.shape[0] == 2
