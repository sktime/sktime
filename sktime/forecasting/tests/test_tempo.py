"""Tests for TEMPOForecaster."""

import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.tempo import TEMPOForecaster


class _FakeTEMPO:
    last_pred_length = None

    @classmethod
    def load_pretrained_model(cls, device, repo_id, filename, cache_dir):
        return cls()

    def predict(self, x, pred_length=96):
        self.__class__.last_pred_length = pred_length
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            arr = arr.ravel()
        base = float(arr[-1]) if arr.size else 0.0
        return np.asarray([base + i for i in range(pred_length)], dtype=float)


@pytest.fixture
def fake_tempo(monkeypatch):
    """Install a fake TEMPO module at the import boundary."""
    monkeypatch.setattr(
        "sktime.forecasting.base._base._check_estimator_deps",
        lambda *args, **kwargs: True,
    )
    fake_tempo_module = types.ModuleType("tempo")
    fake_models_module = types.ModuleType("tempo.models")
    fake_model_module = types.ModuleType("tempo.models.TEMPO")
    fake_model_module.TEMPO = _FakeTEMPO
    fake_models_module.TEMPO = fake_model_module
    fake_tempo_module.models = fake_models_module

    monkeypatch.setitem(sys.modules, "tempo", fake_tempo_module)
    monkeypatch.setitem(sys.modules, "tempo.models", fake_models_module)
    monkeypatch.setitem(sys.modules, "tempo.models.TEMPO", fake_model_module)


def test_tempo_tags_and_get_test_params():
    params = TEMPOForecaster.get_test_params()
    assert isinstance(params, dict)
    assert params["model_path"] == "Melady/TEMPO"

    tags = TEMPOForecaster._tags
    assert tags["authors"]
    assert tags["capability:multivariate"] is False
    assert tags["capability:exogenous"] is False
    assert tags["capability:missing_values"] is False


def test_tempo_point_prediction(fake_tempo):
    y = pd.Series(np.arange(10.0), index=pd.RangeIndex(10))
    forecaster = TEMPOForecaster(model_path="Melady/TEMPO", device="cpu")
    forecaster.fit(y, fh=[1, 2, 3])
    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.Series)
    assert list(y_pred.index) == [10, 11, 12]
    assert np.allclose(y_pred.to_numpy(), np.array([9.0, 10.0, 11.0]))


def test_tempo_relative_and_absolute_fh(fake_tempo):
    y = pd.Series(np.arange(12.0), index=pd.RangeIndex(12))

    forecaster = TEMPOForecaster(model_path="Melady/TEMPO", device="cpu")
    forecaster.fit(y, fh=[2, 4])
    pred_rel = forecaster.predict()
    assert list(pred_rel.index) == [13, 15]
    assert _FakeTEMPO.last_pred_length == 4
    assert np.allclose(pred_rel.to_numpy(), np.array([12.0, 14.0]))

    abs_fh = ForecastingHorizon([14, 16], is_relative=False)
    forecaster = TEMPOForecaster(model_path="Melady/TEMPO", device="cpu")
    forecaster.fit(y, fh=abs_fh)
    pred_abs = forecaster.predict()
    assert list(pred_abs.index) == [14, 16]
    assert _FakeTEMPO.last_pred_length == 5
    assert np.allclose(pred_abs.to_numpy(), np.array([13.0, 15.0]))


def test_tempo_serialization_reuses_cached_model(fake_tempo):
    y = pd.Series(np.arange(10.0), index=pd.RangeIndex(10))
    forecaster = TEMPOForecaster(model_path="Melady/TEMPO", device="cpu")
    forecaster.fit(y, fh=[1, 2, 3])

    restored = pickle.loads(pickle.dumps(forecaster))
    assert restored.model_ is None

    y_pred = restored.predict()

    assert restored.model_ is forecaster.model_
    assert list(y_pred.index) == [10, 11, 12]
    assert np.allclose(y_pred.to_numpy(), np.array([9.0, 10.0, 11.0]))
