# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for seasonality parameter estimators."""

__author__ = ["fkiraly"]

import numpy as np
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.datasets import load_airline
from sktime.param_est.seasonality import (
    SeasonalityACF,
    SeasonalityACFqstat,
    SeasonalityCanovaHansen,
)


def _make_stable_seasonal_series(sp=12, n_obs=240, random_state=12345):
    """Generate a deterministic seasonal series with stable seasonality."""
    rng = np.random.default_rng(random_state)
    t = np.arange(n_obs) + 1
    signal = 2.0 * np.sin(2 * np.pi * t / sp)
    signal = signal + 0.3 * np.cos(4 * np.pi * t / sp)
    noise = rng.normal(scale=0.2, size=n_obs)
    return signal + noise


def _make_seasonal_unit_root_series(sp=12, n_obs=480, random_state=0):
    """Generate a series with a seasonal unit root at period ``sp``."""
    rng = np.random.default_rng(random_state)
    y = np.zeros(n_obs)
    innovations = rng.normal(scale=1.0, size=n_obs)

    for i in range(sp, n_obs):
        y[i] = y[i - sp] + innovations[i]

    return y


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
    from sktime.transformations.difference import Differencer

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


def test_seasonality_canova_hansen_stable_seasonality():
    """Test Canova-Hansen on a stable seasonal process."""
    X = _make_stable_seasonal_series()
    sp_est = SeasonalityCanovaHansen(sp=12)
    sp_est.fit(X)

    fitted = sp_est.get_fitted_params()
    assert fitted["sp"] == 12
    assert fitted["D"] == 0
    assert fitted["seasonal_unit_root"] is False
    assert fitted["critical_value"] == pytest.approx(2.7391007)
    assert fitted["test_statistic"] < fitted["critical_value"]


def test_seasonality_canova_hansen_seasonal_unit_root():
    """Test Canova-Hansen on a seasonal unit root process."""
    X = _make_seasonal_unit_root_series()
    sp_est = SeasonalityCanovaHansen(sp=12)
    sp_est.fit(X)

    fitted = sp_est.get_fitted_params()
    assert fitted["sp"] == 12
    assert fitted["D"] == 1
    assert fitted["seasonal_unit_root"] is True
    assert fitted["test_statistic"] > fitted["critical_value"]
