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
