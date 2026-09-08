"""Tests for TafsutForecaster."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.tafsut import TafsutForecaster


class _FakeConfig:
    quantiles = tuple(np.arange(1, 10) / 10)


class _FakeModel:
    cfg = _FakeConfig()


def _make_forecaster():
    forecaster = TafsutForecaster(local_files_only=True)
    forecaster._load_model = _FakeModel
    return forecaster


def _fit_forecaster(forecaster, y):
    with patch(
        "sktime.forecasting.base._base._check_estimator_deps",
        return_value=True,
    ):
        forecaster.fit(y)


def _fake_forecast(model, horizon):
    quantiles = np.asarray(model.cfg.quantiles)
    values = np.broadcast_to(quantiles, (1, horizon, len(quantiles))).copy()
    values += np.arange(horizon)[None, :, None]
    return values


def test_tafsut_tags_and_test_params():
    forecaster = TafsutForecaster()

    assert forecaster.get_tag("capability:exogenous") is False
    assert forecaster.get_tag("capability:multivariate") is False
    assert forecaster.get_tag("capability:pred_int") is True
    assert forecaster.get_test_params()[0]["model_path"] is None


def test_tafsut_point_prediction_and_horizon():
    y = pd.Series(np.arange(10.0), name="y")
    forecaster = _make_forecaster()
    forecaster._forecast = lambda horizon: _fake_forecast(_FakeModel(), horizon)
    _fit_forecaster(forecaster, y)

    y_pred = forecaster.predict(fh=[1, 3, 5])

    expected_index = pd.RangeIndex(10, 15, 2)
    np.testing.assert_array_equal(y_pred.index, expected_index)
    np.testing.assert_allclose(y_pred.to_numpy(), [0.5, 2.5, 4.5])


def test_tafsut_quantiles_preserve_requested_order_and_interpolate():
    y = pd.Series(np.arange(10.0), name="y")
    forecaster = _make_forecaster()
    forecaster._forecast = lambda horizon: _fake_forecast(_FakeModel(), horizon)
    _fit_forecaster(forecaster, y)

    result = forecaster._predict_quantiles(
        ForecastingHorizon([1, 2]), None, [0.8, 0.25, 0.8]
    )

    expected_columns = pd.MultiIndex.from_tuples([("y", 0.8), ("y", 0.25), ("y", 0.8)])
    assert result.columns.equals(expected_columns)
    np.testing.assert_allclose(result.iloc[0].to_numpy(), [0.8, 0.25, 0.8])
    np.testing.assert_allclose(result.iloc[1].to_numpy(), [1.8, 1.25, 1.8])


def test_tafsut_absolute_horizon_and_missing_context():
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    y = pd.Series(np.arange(10.0), index=index, name="y")
    y.iloc[3] = np.nan
    forecaster = _make_forecaster()
    forecaster._forecast = lambda horizon: _fake_forecast(_FakeModel(), horizon)
    _fit_forecaster(forecaster, y)
    future_index = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=3, freq="D")

    result = forecaster.predict(fh=future_index[[0, 2]])

    assert result.index.equals(future_index[[0, 2]])
    np.testing.assert_allclose(result.to_numpy(), [0.5, 2.5])


def test_tafsut_clamps_quantiles_outside_native_range():
    forecaster = _make_forecaster()
    forecaster._forecast = lambda horizon: _fake_forecast(_FakeModel(), horizon)
    _fit_forecaster(forecaster, pd.Series(np.arange(10.0), name="y"))

    result = forecaster.predict_quantiles(fh=[1], alpha=[0.05, 0.95])

    np.testing.assert_allclose(result.iloc[0].to_numpy(), [0.1, 0.9])
