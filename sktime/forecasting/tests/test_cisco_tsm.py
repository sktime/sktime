# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for CiscoTSMForecaster.

CiscoTSMForecaster has a restrictive dependency set,
which may cancel out with the matrix testing strategy.

Therefore, we call a check_estimator separately.
"""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.cisco_tsm import CiscoTSMForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(CiscoTSMForecaster),
    reason="run test only if softdeps are present",
)
def test_cisco_tsm_forecaster_predict_proba():
    """Verify that predict_proba is consistent with predict_quantiles."""
    from skpro.distributions import HistogramQPD

    # Create a simple test series
    y = pd.Series([10.0, 12.0, 15.0, 13.0, 20.0, 18.0, 22.0, 25.0, 23.0, 28.0])

    forecaster = CiscoTSMForecaster()
    forecaster.fit(y)

    fh = [1, 2]
    alpha = [0.1, 0.5, 0.9]

    # Predict quantiles directly
    q_direct = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    # Predict probability and extract quantiles from the distribution
    pred_dist = forecaster.predict_proba(fh=fh)
    assert isinstance(pred_dist, HistogramQPD)
    q_from_proba = pred_dist.quantile(alpha)

    # Directly compare the outputs
    np.testing.assert_allclose(q_direct.values, q_from_proba.values, atol=1e-5)


def test_cisco_tsm_update_restricts_context_to_context_length():
    """``_update`` retains only the latest ``context_length`` observations."""
    y = pd.Series(np.arange(20, dtype=np.float32))
    forecaster = CiscoTSMForecaster(context_length=4, ignore_deps=True)
    forecaster._fit(y)

    y_new = pd.Series(
        np.arange(20, 25, dtype=np.float32), index=pd.RangeIndex(20, 25)
    )
    forecaster._update(y_new)

    assert len(forecaster._context_) == 4
    np.testing.assert_array_equal(forecaster._context_, [21, 22, 23, 24])


def test_cisco_tsm_update_no_context_length_keeps_all():
    """With ``context_length=None`` all appended history is retained."""
    y = pd.Series(np.arange(8, dtype=np.float32))
    forecaster = CiscoTSMForecaster(context_length=None, ignore_deps=True)
    forecaster._fit(y)

    y_new = pd.Series(
        np.arange(8, 11, dtype=np.float32), index=pd.RangeIndex(8, 11)
    )
    forecaster._update(y_new)

    assert len(forecaster._context_) == 11
    np.testing.assert_array_equal(forecaster._context_, np.arange(11, dtype=np.float32))

