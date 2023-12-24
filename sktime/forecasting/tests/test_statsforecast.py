"""Tests for StatsForecast.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.statsforecast import StatsForecastMSTL
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(StatsForecastMSTL),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@patch("statsforecast.models.AutoETS", autospec=True)
def test_statsforecast_mstl(mock_autoets):
    """
    Check that StatsForecast MSTL adapter calls trend forecaster with
    the correct arguments.
    """
    from sktime.datasets import load_airline

    y = load_airline()

    predict = mock_autoets.return_value.predict
    predict.return_value = {
        "mean": np.arange(36, dtype=np.float64),
        "lo-95.0": np.arange(36, dtype=np.float64),
        "hi-95.0": np.arange(36, dtype=np.float64),
    }

    model = StatsForecastMSTL(season_length=[12])
    model.fit(y)
    fh_index = pd.PeriodIndex(pd.date_range("1961-01", periods=36, freq="M"))
    fh = ForecastingHorizon(fh_index, is_relative=False)
    model.predict_interval(fh, coverage=0.95)
    predict.assert_called_with(36, X=None, level=[95.0])
