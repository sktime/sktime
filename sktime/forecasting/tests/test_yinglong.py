"""Tests for the YingLong forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.yinglong import YingLongForecaster
from sktime.tests.test_switch import run_test_for_class


class _FakeYingLongModel:
    """Small fake YingLong model for testing the sktime wrapper."""

    class _Config:
        patch_size = 4
        block_size = 16

    config = _Config()

    def generate(self, past_values, future_token):
        """Return deterministic predictions for the requested horizon."""
        import torch

        values = torch.arange(
            1,
            future_token + 1,
            dtype=torch.float32,
            device=past_values.device,
        )

        return values.reshape(1, future_token, 1)


def _make_fake_forecaster():
    """Create a YingLong forecaster with a fake pretrained model."""
    forecaster = YingLongForecaster(
        device="cpu",
        torch_dtype="float32",
        ignore_deps=True,
    )

    forecaster._load_model = lambda: _FakeYingLongModel()

    return forecaster


def test_yinglong_constructor_and_tags():
    """Test constructor parameters and estimator tags."""
    forecaster = YingLongForecaster()

    assert forecaster.model_path == "qcw2333/YingLong_6m"
    assert forecaster.device == "cuda"
    assert forecaster.torch_dtype == "bfloat16"
    assert forecaster.trust_remote_code is True
    assert forecaster.ignore_deps is False

    assert forecaster.get_tag("tests:vm") is True
    assert forecaster.get_tag("capability:multivariate") is False
    assert forecaster.get_tag("capability:exogenous") is False


def test_yinglong_ignore_deps_clears_dependencies():
    """Test that ignore_deps removes optional dependency requirements."""
    forecaster = YingLongForecaster(ignore_deps=True)

    assert forecaster.get_tag("python_dependencies") == []


def test_yinglong_fit_predict_with_fake_model():
    """Test fit and predict without loading the real YingLong model."""
    y = pd.Series(
        np.arange(8, dtype=np.float32),
        index=pd.RangeIndex(8),
        name="y",
    )

    forecaster = _make_fake_forecaster()
    

    fh = ForecastingHorizon([1, 3, 5], is_relative=True)

    forecaster._fit(y)
    forecaster._set_cutoff(y.index[-1])
    y_pred = forecaster._predict(fh)

    np.testing.assert_array_equal(
        y_pred.to_numpy(),
        np.array([1.0, 3.0, 5.0], dtype=np.float32),
    )

    assert list(y_pred.index) == [8, 10, 12]
    assert y_pred.name == "y"


def test_yinglong_rejects_non_multiple_context_length():
    """Test that context length must be divisible by patch size."""
    y = pd.Series(
        np.arange(6, dtype=np.float32),
        index=pd.RangeIndex(6),
    )

    forecaster = _make_fake_forecaster()

    with pytest.raises(
        ValueError,
        match="multiple of the model patch size",
    ):
        forecaster._fit(y)


def test_yinglong_rejects_too_short_context():
    """Test that context shorter than one patch is rejected."""
    y = pd.Series(
        np.arange(3, dtype=np.float32),
        index=pd.RangeIndex(3),
    )

    forecaster = _make_fake_forecaster()

    with pytest.raises(
        ValueError,
        match="requires at least",
    ):
        forecaster._fit(y)


def test_yinglong_rejects_insample_forecast():
    """Test that YingLong only supports out-of-sample forecasting."""
    y = pd.Series(
        np.arange(8, dtype=np.float32),
        index=pd.RangeIndex(8),
    )

    forecaster = _make_fake_forecaster()
    forecaster._fit(y)

    fh = ForecastingHorizon([-1, 1], is_relative=True)

    with pytest.raises(
        ValueError,
        match="only supports out-of-sample",
    ):
        forecaster._predict(fh)


def test_yinglong_rejects_forecast_exceeding_model_capacity():
    """Test that forecasts exceeding model capacity are rejected."""
    y = pd.Series(
        np.arange(8, dtype=np.float32),
        index=pd.RangeIndex(8),
    )

    forecaster = _make_fake_forecaster()
    forecaster._fit(y)

    fh = ForecastingHorizon([1, 9], is_relative=True)

    with pytest.raises(
        ValueError,
        match="maximum sequence length",
    ):
        forecaster._predict(fh)


@pytest.mark.skipif(
    not run_test_for_class(YingLongForecaster),
    reason=(
        "run test only if YingLong soft dependencies are present "
        "and the test is selected"
    ),
)
def test_yinglong_airline_predictions():
    """Run an end-to-end smoke test with the real YingLong model."""
    y = load_airline()
    y_train = y.iloc[:-12]
    fh = np.arange(1, 13)

    forecaster = YingLongForecaster()

    y_pred = forecaster.fit(y_train).predict(fh=fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == 12
    assert np.isfinite(y_pred.to_numpy()).all()