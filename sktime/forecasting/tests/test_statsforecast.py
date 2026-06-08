"""Tests for StatsForecast.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.statsforecast import StatsForecastAutoCES, StatsForecastMSTL
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

    if not model.get_tag("capability:pred_int"):
        return

    model.fit(y)
    fh_index = pd.PeriodIndex(pd.date_range("1961-01", periods=36, freq="M"))
    fh = ForecastingHorizon(fh_index, is_relative=False)
    model.predict_interval(fh, coverage=0.95)
    predict.assert_called_with(36, X=None, level=[95.0])


@pytest.mark.skipif(
    not run_test_for_class(StatsForecastMSTL),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "fh",
    [[1, 2, 3], [1], 0, 5, None],
    ids=["valid fh", "different fh predict", "in-sample", "scalar", "None fh"],
)
def test_statsforecast_mstl_with_fh(request, fh):
    """
    Check that StatsForecast MSTL adapter calls trend forecaster with
    the correct arguments.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import make_reduction

    y = load_airline()

    regressor = GradientBoostingRegressor()
    reduction_forecaster = make_reduction(
        regressor, window_length=15, strategy="direct"
    )

    model = StatsForecastMSTL(
        season_length=[3, 12], trend_forecaster=reduction_forecaster
    )

    try:
        # fit with fh passed to model
        model.fit(y, fh=fh)
    except NotImplementedError:
        assert "in-sample" in request.node.name, (
            "Unexpected exception raised - should have failed with "
            "NotImplementedError, DirectTabularRegressionForecaster "
            "can not perform in-sample prediction ..."
        )
        return
    except ValueError:
        assert "None fh" in request.node.name, (
            "Unexpected exception raised - should have failed with ValueError, "
            "The forecasting horizon `fh` must be passed to `fit` of ..."
        )
        return
    if "different fh predict" in request.node.name:
        # predict with different fh
        fh.append(2)
    try:
        model.predict(fh=fh)
    except ValueError:
        assert "different fh predict" in request.node.name, (
            "Unexpected exception raised - should have failed with ValueError, "
            "A different forecasting horizon `fh` has been "
            "provided from "
        )


@pytest.mark.skipif(
    not run_test_for_class(StatsForecastAutoCES),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_constant_series_prediction_with_ces() -> None:
    """Test that StatsForecastAutoCES can predict constant series."""
    dummy_past_data = pd.Series([2] * 10)

    ces_model = StatsForecastAutoCES()
    _ = ces_model.fit(dummy_past_data)

    dummy_future_predictions = ces_model.predict(fh=range(5))

    # StatsForecast switches from CES to Naive if the series is constant
    assert np.all(dummy_future_predictions == 2)
