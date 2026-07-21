"""Tests the ARCH model"""

__author__ = ["SzymonStolarski"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arch import ARCH
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ARCH),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARCH_against_arch_model():
    """
    Compare sktime's ARCH implementation against direct arch_model from arch package.

    Notes
    -----
    * Predicts variance using underlying estimator and the wrapper.
    * Compares values starting from fh[2] to test if the predictions are assigned
        appropriately.
    """
    from arch import arch_model

    # Generate a dataset mimicking returns data on which ARCH is typically applied
    date_range = pd.date_range(start="2020-01-03", end="2025-06-20", freq="B")
    sample_returns_series = pd.Series(
        np.random.normal(loc=0, scale=0.02, size=len(date_range)),
        index=pd.PeriodIndex(date_range, freq="D"),
    )

    sktime_model = ARCH(
        p=1,
        mean="Zero",
        dist="Normal",
        vol="ARCH",
        method="analytic",
        random_state=42,
        rescale=False,
    )
    sktime_model.fit(sample_returns_series)
    # Predict variance, starting from fh[2]
    sktime_pred = sktime_model.predict_var(fh=[i for i in range(2, 6)])

    arch_model_arch = arch_model(
        sample_returns_series,
        p=1,
        mean="Zero",
        dist="Normal",
        vol="ARCH",
        rescale=False,
    )
    arch_model_arch_fitted = arch_model_arch.fit(disp="off")
    arch_model_arch_pred = arch_model_arch_fitted.forecast(horizon=5)
    # Extract the variance predictions, starting from fh[2]
    arch_model_arch_pred = arch_model_arch_pred.variance.iloc[0][1:]

    assert np.allclose(sktime_pred.values, arch_model_arch_pred.values)


@pytest.mark.skipif(
    not run_test_for_class(ARCH),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARCH_insample():
    """Test if in-sample prediction raises an error and aligns with the arch package."""
    from arch import arch_model

    # Generate a dataset mimicking returns data on which ARCH is typically applied
    date_range = pd.date_range(start="2020-01-03", end="2025-06-20", freq="D")
    sample_returns_series = pd.Series(
        np.random.normal(loc=0, scale=0.02, size=len(date_range)),
        index=pd.PeriodIndex(date_range, freq="D"),
    )

    sktime_model = ARCH(
        p=1,
        mean="AR",
        dist="Normal",
        vol="ARCH",
        method="analytic",
        random_state=42,
        rescale=False,
    )
    sktime_model.fit(sample_returns_series)
    sktime_pred_mean = sktime_model.predict(
        fh=[i for i in range(-len(sample_returns_series) + 1, 1)]
    )
    sktime_pred_var = sktime_model.predict_var(
        fh=[i for i in range(-len(sample_returns_series) + 1, 1)]
    )

    arch_model_arch = arch_model(
        sample_returns_series,
        p=1,
        mean="AR",
        dist="Normal",
        vol="ARCH",
        rescale=False,
    )
    arch_model_arch_fitted = arch_model_arch.fit(disp="off")
    arch_model_arch_pred_mean = sample_returns_series - arch_model_arch_fitted.resid
    arch_model_arch_pred_var = (arch_model_arch_fitted.conditional_volatility) ** 2

    assert np.allclose(
        sktime_pred_mean.values, arch_model_arch_pred_mean.values
    ) and np.allclose(sktime_pred_var.values, arch_model_arch_pred_var.values)


@pytest.mark.skipif(
    not run_test_for_class(ARCH),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ARCH_with_exogenous():
    """Test ARX-GARCH with exogenous variables against arch package.

    This test verifies that exogenous variables are correctly passed through
    to the underlying arch package when using ARX or HARX mean models.
    Regression test for bug where super().__init__() reset dynamic tags.
    """
    from arch import arch_model

    # Generate returns data and exogenous variable
    np.random.seed(42)
    n_train = 100
    n_forecast = 3
    
    date_range = pd.date_range(start="2020-01-03", periods=n_train + n_forecast, freq="B")
    y = pd.Series(
        np.random.normal(loc=0, scale=0.02, size=n_train),
        index=pd.PeriodIndex(date_range[:n_train], freq="D"),
    )
    # Create exogenous variable for full period
    X_full = pd.DataFrame(
        {"exog": np.random.normal(loc=0, scale=0.01, size=n_train + n_forecast)},
        index=pd.PeriodIndex(date_range, freq="D"),
    )
    X_train = X_full.iloc[:n_train]
    X_forecast = X_full.iloc[n_train : n_train + n_forecast]

    # Fit sktime ARX-GARCH model
    sktime_model = ARCH(
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="Normal",
        method="analytic",
        random_state=42,
        rescale=False,
    )
    sktime_model.fit(y, X=X_train)
    
    # Verify that exogenous capability is enabled
    assert sktime_model.get_tag("capability:exogenous") is True
    
    # Predict with exogenous variables (X_forecast must match horizon exactly)
    sktime_pred = sktime_model.predict(fh=[1, 2, 3], X=X_forecast)

    # Fit arch package model directly for comparison
    arch_model_direct = arch_model(
        y,
        x=X_train,
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="Normal",
        rescale=False,
    )
    arch_fitted = arch_model_direct.fit(disp="off")
    
    # Forecast with arch package
    x_forecast_dict = {col: X_forecast[col].values for col in X_forecast.columns}
    arch_pred = arch_fitted.forecast(
        horizon=3, x=x_forecast_dict, method="analytic"
    )

    # Compare predictions - should be very close
    assert np.allclose(sktime_pred.values, arch_pred.mean.iloc[-1, :3].values, rtol=1e-5)
