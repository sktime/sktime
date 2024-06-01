# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for seasonality parameter estimators."""

__author__ = ["fkiraly"]

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
