# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for impulse response function estimators"""

__author__ = ["OldPatrick"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.param_est.impulse import ImpulseResponseFunction
from sktime.utils.dependencies import _check_estimator_deps

# VARMAX tests


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_varmax():
    """Test ImpulseResponseFunctions on airline data with VARMAX."""
    from sktime.forecasting.varmax import VARMAX as skmax

    # Convergence and estimation warnings happen regularly in statsmodels too.
    np.random.seed(42)

    X1 = load_airline().values.astype(float)
    X1_stationary = np.diff(np.log(X1))
    noise = np.random.normal(scale=0.05, size=len(X1_stationary))
    X2_stationary = 0.6 * X1_stationary + 0.4 * np.roll(X1_stationary, 1) + noise

    df = pd.DataFrame({"X1": X1_stationary[1:], "X2": X2_stationary[1:]})
    df.index = pd.date_range("1949-02-01", periods=len(df), freq="MS")

    sk_model = skmax(order=(1, 2), trend="c").fit(df)
    sk_res = ImpulseResponseFunction(sk_model)
    sk_res.fit(df)

    actual = np.round(sk_res.get_fitted_params()["irf"].sum())
    expected = 1.0

    assert actual == expected


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_additional_irfparams_on_varmax():
    """Test more ImpulseResponseFunction parameters with airline data on VARMAX."""
    from sktime.forecasting.varmax import VARMAX as skmax

    # Convergence and estimation warnings happen regularly in statsmodels too.
    np.random.seed(42)

    X1 = load_airline().values.astype(float)
    X1_stationary = np.diff(np.log(X1))
    noise = np.random.normal(scale=0.05, size=len(X1_stationary))
    X2_stationary = 0.6 * X1_stationary + 0.4 * np.roll(X1_stationary, 1) + noise

    df = pd.DataFrame({"X1": X1_stationary[1:], "X2": X2_stationary[1:]})
    df.index = pd.date_range("1949-02-01", periods=len(df), freq="MS")

    sk_model = skmax(order=(1, 2), trend="c").fit(df)
    sk_res = ImpulseResponseFunction(sk_model, cumulative=True, steps=4)
    sk_res.fit(df)

    actual = np.round(sk_res.get_fitted_params()["irf"].sum())
    expected = 0

    assert actual == expected


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_vmax_against_statsmodels():
    """Compare sktime irf VARMAX wrapper against statsmodels irf"""
    from statsmodels.tsa.statespace.varmax import VARMAX as statsmax

    from sktime.forecasting.varmax import VARMAX as skmax

    # Convergence and estimation warnings happen regularly in statsmodels too.
    np.random.seed(42)

    X1 = load_airline().values.astype(float)
    X1_stationary = np.diff(np.log(X1))
    noise = np.random.normal(scale=0.05, size=len(X1_stationary))
    X2_stationary = 0.6 * X1_stationary + 0.4 * np.roll(X1_stationary, 1) + noise

    df = pd.DataFrame({"X1": X1_stationary[1:], "X2": X2_stationary[1:]})
    df.index = pd.date_range("1949-02-01", periods=len(df), freq="MS")

    stats_model = statsmax(df, order=(1, 2), trend="n")
    fitted_model_stats_vmax = stats_model.fit()
    res_vmax_stats = fitted_model_stats_vmax.impulse_responses()

    sk_model = skmax(order=(1, 2), trend="n").fit(df)
    res_vmax_sk = ImpulseResponseFunction(sk_model)
    res_vmax_sk.fit(df)

    np.testing.assert_array_equal(
        res_vmax_stats, res_vmax_sk.get_fitted_params()["irf"]
    )


# DynamicFactor tests
@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_dynamic():
    """Test ImpulseResponseFunctions on airline data with DynamicFactor."""
    from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn

    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})

    sk_model = skdyn(
        k_factors=1, factor_order=2, error_order=2, enforce_stationarity=False
    ).fit(df)
    sktime_res = ImpulseResponseFunction(sk_model)
    sktime_res.fit(df)

    actual = np.round(sktime_res.get_fitted_params()["irf"].sum())
    expected = 2184

    assert actual == expected


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_additional_irfparams_on_dyn():
    """Test more ImpulseResponseFunction parameters with airline data on DynFac."""
    from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn

    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})

    sk_model = skdyn(k_factors=1, factor_order=2).fit(df)
    sktime_res = ImpulseResponseFunction(sk_model, cumulative=True, steps=2)
    sktime_res.fit(df)

    actual = np.round(sktime_res.get_fitted_params()["irf"].sum())
    expected = 5627.0

    assert actual == expected


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_dyn_against_statsmodels():
    """Compare sktime irf DynamicFactor wrapper against statsmodels irf"""
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor as statsdyn

    from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn

    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})

    st_model = statsdyn(df, k_factors=1, factor_order=2)
    fitted_model = st_model.fit()
    stats_res = fitted_model.impulse_responses(orthogonalized=True)

    sk_model = skdyn(k_factors=1, factor_order=2).fit(df)
    sktime_res = ImpulseResponseFunction(sk_model, orthogonalized=True)
    sktime_res.fit(df)

    np.testing.assert_array_equal(stats_res, sktime_res.get_fitted_params()["irf"])


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_vecm():
    """Test ImpulseResponseFunctions on airline data with VECM."""
    pass


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_vecm_against_statsmodels():
    """Compare sktime irf VECM wrapper against statsmodels irf"""
    pass


print(
    test_irf_on_dynamic(),
    test_additional_irfparams_on_dyn(),
    test_irf_dyn_against_statsmodels(),
    test_irf_on_varmax(),
    test_additional_irfparams_on_varmax(),
    test_irf_vmax_against_statsmodels(),
)
