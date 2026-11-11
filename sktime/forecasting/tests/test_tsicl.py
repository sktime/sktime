# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TSICLForecaster."""

import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.tsicl import TSICLForecaster


def test_tsicl_get_test_params():
    """Test get_test_params returns valid list of dicts."""
    params = TSICLForecaster.get_test_params()
    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, dict) for p in params)


@pytest.mark.skipif(
    not _check_estimator_deps(TSICLForecaster, severity="none"),
    reason="run test only if TSICLForecaster dependencies (tsicl, torch) are present",
)
def test_tsicl_initialization():
    """Test that TSICLForecaster can be initialized."""
    forecaster = TSICLForecaster(model_path="taharnbl/TS-ICL")
    assert forecaster.model_path == "taharnbl/TS-ICL"
    assert forecaster.denormalize is True
