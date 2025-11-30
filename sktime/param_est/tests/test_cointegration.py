# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for johansen parameter estimator."""

__author__ = ["PBormann"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.param_est.cointegration import JohansenCointegration
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(JohansenCointegration, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_cointegration():
    """Test Cointegration on airline data."""
    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})
    coint_est = JohansenCointegration()
    coint_est.fit(df)

    assert list(coint_est.get_fitted_params()["ind"]) == [0, 1]
    actual = coint_est.get_fitted_params()["ind"]
    expected = [0, 1]
    np.testing.assert_array_equal(actual, expected)

@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_against_statsmodels():
    """Compare sktime's cointegrtion wrapper against statsmodels cointegration"""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    X = load_airline()
    X2 = X.shift(1).bfill()
    df = pd.DataFrame({"X": X, "X2": X2})
    coint_est = JohansenCointegration(det_order=0, k_ar_diff=0)
    coint_est.fit(df)

    sktime_coint = coint_est.get_fitted_params()["cvm"]
    statsmodels_coint = coint_johansen(endog=df, det_order=0, k_ar_diff=0).cvm

    np.testing.assert_array_equal(sktime_coint, statsmodels_coint)
