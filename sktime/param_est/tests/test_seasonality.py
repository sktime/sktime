# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for seasonality parameter estimators."""

__author__ = ["fkiraly", "Akanksha Trehun"]

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.param_est.seasonality import SeasonalityACF, SeasonalityACFqstat
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf():
    """Test SeasonalityACF on airline data."""
    X = load_airline().diff()[1:]
    sp_est = SeasonalityACF()
    sp_est.fit(X)

    assert sp_est.get_fitted_params()["sp"] == 12
    actual = sp_est.get_fitted_params()["sp_significant"]
    expected = np.array([12, 11])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf_pipeline():
    """Test SeasonalityACF pipeline on airline data."""
    from sktime.transformations.series.difference import Differencer

    X = load_airline()
    sp_est = Differencer() * SeasonalityACF()
    sp_est.fit(X)
    assert sp_est.get_fitted_params()["sp"] == 12
    actual = sp_est.get_fitted_params()["sp_significant"]
    expected = np.array([12, 11])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACFqstat, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf_qstat():
    """Test SeasonalityACFqstat on airline data."""
    X = load_airline().diff()[1:]
    sp_est = SeasonalityACFqstat(candidate_sp=[3, 7, 12])
    sp_est.fit(X)

    actual = sp_est.get_fitted_params()["sp_significant"]
    expected = np.array([12, 7, 3])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _check_estimator_deps(SeasonalityACFqstat, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_seasonality_acf_qstat_p_adjust_none_consistent():
    """Test SeasonalityACFqstat p_adjust='none' returns significant periods.

    Regression test for bug #10002: reject_cand was inverted when p_adjust='none',
    causing non-significant periods to be selected as seasonal periods.
    """
    X = load_airline().diff()[1:]
    sp_est_adj = SeasonalityACFqstat(candidate_sp=[3, 7, 12], p_adjust="fdr_bh")
    sp_est_none = SeasonalityACFqstat(candidate_sp=[3, 7, 12], p_adjust="none")
    sp_est_adj.fit(X)
    sp_est_none.fit(X)

    # Both variants should agree that period 12 is strongly seasonal
    assert 12 in sp_est_none.sp_significant_
    # p_adjust='none' primary sp should match adjusted variant on clear seasonal data
    assert sp_est_none.sp_ == sp_est_adj.sp_
