# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for CiscoTSMForecaster.

CiscoTSMForecaster has a restrictive dependency set,
which may cancel out with the matrix testing strategy.

Therefore, we call a check_estimator separately.
"""

import pytest

from sktime.forecasting.cisco_tsm import CiscoTSMForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils import check_estimator


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    import skbase.utils.dependencies
    import skbase.utils.dependencies._dependencies

    import sktime.forecasting.base._base
    import sktime.utils.dependencies

    mock_func = lambda *args, **kwargs: True

    monkeypatch.setattr(
        sktime.forecasting.base._base, "_check_estimator_deps", mock_func
    )
    monkeypatch.setattr(
        sktime.utils.dependencies, "_check_estimator_deps", mock_func
    )
    monkeypatch.setattr(
        skbase.utils.dependencies, "_check_estimator_deps", mock_func
    )
    monkeypatch.setattr(
        skbase.utils.dependencies._dependencies,
        "_check_estimator_deps",
        mock_func,
    )


@pytest.mark.skipif(
    not run_test_for_class(CiscoTSMForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cisco_tsm_forecaster():
    """Run standard test suite for CiscoTSMForecaster."""
    check_estimator(CiscoTSMForecaster, raise_exceptions=True)


def test_cisco_tsm_forecaster_dummy():
    """Run check_estimator using dummy model for compliance testing."""
    # This always runs to verify the dummy model path.
    # Note: CiscoTSMForecaster defaults to ignore_deps=True in get_test_params.
    results = check_estimator(CiscoTSMForecaster, raise_exceptions=False)

    # Check that all non-skipped tests specifically for predict_proba passed
    for test_name, status in results.items():
        if "predict_proba" in test_name:
            assert (
                status == "PASSED"
                or "skipped" in str(status).lower()
                or status == "SKIPPED"
            )


def test_cisco_tsm_forecaster_predict_proba():
    """Verify that predict_proba returns a valid HistogramQPD distribution."""
    import numpy as np
    import pandas as pd
    from skpro.distributions import HistogramQPD

    from sktime.forecasting.cisco_tsm import CiscoTSMForecaster

    # Create simple series
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    forecaster = CiscoTSMForecaster(ignore_deps=True)
    forecaster.fit(y)

    # predict_proba for fh = [1, 2]
    pred_dist = forecaster.predict_proba(fh=[1, 2])

    # Check that it returns a HistogramQPD
    assert isinstance(pred_dist, HistogramQPD)

    # Check shape/index
    assert len(pred_dist.index) == 2
    assert len(pred_dist.columns) == 1

    # Check values
    # For dummy model, predict_proba should return values corresponding to the
    # dummy constant forecast. We request quantiles to verify consistency.
    quantiles_df = pred_dist.quantile([0.1, 0.5, 0.9])

    # Check that the values are correctly returned as the dummy fill (0.0 by default)
    np.testing.assert_allclose(quantiles_df.values, 0.0, atol=1e-5)
