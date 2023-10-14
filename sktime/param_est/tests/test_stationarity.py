# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for stationarity parameter estimators."""

__author__ = ["fkiraly"]

import pytest

from sktime.datasets import load_airline
from sktime.param_est.stationarity import StationarityADF, StationarityKPSS
from sktime.utils.validation._dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(StationarityADF, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_stationarity_adf():
    """Test StationarityADF on airline data, identical to docstring example."""
    X = load_airline()
    sty_est = StationarityADF()
    sty_est.fit(X)
    assert not sty_est.get_fitted_params()["stationary"]


@pytest.mark.skipif(
    not _check_estimator_deps(StationarityKPSS, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_stationarity_kpss():
    """Test StationarityKPSS on airline data, identical to docstring example."""
    X = load_airline()
    sty_est = StationarityKPSS()
    sty_est.fit(X)
    assert not sty_est.get_fitted_params()["stationary"]
