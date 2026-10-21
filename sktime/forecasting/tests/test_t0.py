"""Tests for T0Forecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.t0 import T0Forecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(T0Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies(
        "tfc-t0", "torch", "einops", "rotary-embedding-torch", severity="none"
    ),
    reason="skip test if tfc-t0 dependencies are not installed",
)
def test_t0_forecaster_predict_univariate():
    """Test univariate point and quantile forecast with local random model."""
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.Series(np.arange(20, dtype=float), index=index, name="val")
    forecaster = T0Forecaster(model_path=None, license_accepted=True)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2, 3])
    assert len(y_pred) == 3

    y_quantiles = forecaster.predict_quantiles(fh=[1, 2, 3], alpha=[0.1, 0.5, 0.9])
    assert y_quantiles.shape == (3, 3)


@pytest.mark.skipif(
    not run_test_for_class(T0Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies(
        "tfc-t0", "torch", "einops", "rotary-embedding-torch", severity="none"
    ),
    reason="skip test if tfc-t0 dependencies are not installed",
)
def test_t0_forecaster_predict_exogenous():
    """Test forecast with exogenous variables."""
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.Series(np.arange(20, dtype=float), index=index, name="val")
    X = pd.DataFrame(
        {"cov1": np.arange(20, dtype=float), "cov2": np.ones(20)}, index=index
    )

    y_tr = y.iloc[:15]
    X_tr, X_te = X.iloc[:15], X.iloc[15:]

    forecaster = T0Forecaster(model_path=None, license_accepted=True)
    forecaster.fit(y_tr, X=X_tr)
    y_pred = forecaster.predict(fh=[1, 2, 3], X=X_te)
    assert len(y_pred) == 3


@pytest.mark.skipif(
    not run_test_for_class(T0Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies(
        "tfc-t0", "torch", "einops", "rotary-embedding-torch", severity="none"
    ),
    reason="skip test if tfc-t0 dependencies are not installed",
)
def test_t0_forecaster_predict_multivariate():
    """Test multivariate forecast with local random model."""
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    y = pd.DataFrame(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) * 2,
        },
        index=index,
    )
    forecaster = T0Forecaster(model_path=None, license_accepted=True)
    forecaster.fit(y)
    y_pred = forecaster.predict(fh=[1, 2])
    assert y_pred.shape == (2, 2)
