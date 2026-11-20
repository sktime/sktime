"""Tests for Arps decline curve analysis forecasters."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arps_dca import ArpsExponential
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ArpsExponential),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_decline_rate_per_day_for_any_datetime_resolution(unit):
    """Test the fitted decline rate is per day, whatever the datetime resolution."""
    start = np.datetime64("2020-01-01", unit)
    index = pd.DatetimeIndex(start + np.arange(30).astype("timedelta64[D]"))
    y = pd.Series(1000 * np.exp(-0.01 * np.arange(30)), index=index)

    forecaster = ArpsExponential().fit(y)

    np.testing.assert_allclose(forecaster.get_fitted_params()["Di"], 0.01, rtol=1e-6)
