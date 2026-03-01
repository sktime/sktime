# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for impulse response function estimators"""

__author__ = ["OldPatrick"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.param_est.impulse import ImpulseResponseFunction
from sktime.utils.dependencies import _check_estimator_deps

@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_varmax():
    """Test ImpulseResponseFunctions on airline data with VARMAX."""
    pass
    #assert list(coint_est.get_fitted_params()["ind"]) == [0, 1]
    # actual = coint_est.get_fitted_params()["ind"]
    # expected = [0, 1]
    # np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_vmax_against_statsmodels():
    """Compare sktime irf VARMAX wrapper against statsmodels irf"""
    X1 = load_airline().values.astype(float)
    X1_stationary = np.diff(np.log(X1))
    np.random.seed(42)
    noise = np.random.normal(scale=0.05, size=len(X1_stationary))
    X2_stationary = 0.6 * X1_stationary + 0.4 * np.roll(X1_stationary, 1) + noise
    df2 = pd.DataFrame({
        "X1": X1_stationary[1:],
        "X2": X2_stationary[1:]
    })
    df2.index = pd.date_range("1949-02-01", periods=len(df2), freq="MS")

    st_model2 = statsmax(df2, order=(1, 2), trend ="c")
    fitted_model2 = st_model2.fit()
    stats_res2 = fitted_model2.impulse_responses()
    print(stats_res2)

    sk_model2 = skmax(order=(1, 2), trend ="c").fit(df2)
    sktime_res2 = ImpulseResponseFunction(sk_model2)
    sktime_res2.fit(df2)
    print(sktime_res2.get_fitted_params()["irf"])


@pytest.mark.skipif(
    not _check_estimator_deps(ImpulseResponseFunction, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_irf_on_dynamic():
    """Test ImpulseResponseFunctions on airline data with DynamicFactor."""
    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})
    irf_est = ImpulseResponseFunction()
    irf_est.fit(df)

    #assert list(coint_est.get_fitted_params()["ind"]) == [0, 1]
    # actual = coint_est.get_fitted_params()["ind"]
    # expected = [0, 1]
    # np.testing.assert_array_equal(actual, expected)


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


test_irf_dyn_against_statsmodels()