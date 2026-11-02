"""Tests for the Tabby forecaster adapter."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.tabby import TabbyForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if torch and transformers are installed",
)


class _FakeTabbyModel:
    """Small deterministic stand-in for the pretrained model."""

    config = type("Config", (), {"context_length": 8192, "patch_size": 16})()

    def __init__(self):
        self.contexts = []
        self.horizons = []

    def predict(self, context, prediction_length):
        import torch

        self.contexts.append(context.detach().cpu().numpy().copy())
        self.horizons.append(prediction_length)
        quantiles = torch.arange(99, dtype=torch.float32).reshape(1, 99, 1) * 100
        steps = torch.arange(prediction_length, dtype=torch.float32).reshape(1, 1, -1)
        return quantiles + steps


def _fit_fake(monkeypatch, y):
    """Fit a forecaster backed by the deterministic fake model."""
    model = _FakeTabbyModel()
    monkeypatch.setattr(TabbyForecaster, "_load_model", lambda self: model)
    forecaster = TabbyForecaster(trust_remote_code=True).fit(y)
    return forecaster, model


def test_tabby_predict_horizon_and_non_contiguous_fh(monkeypatch):
    """Prediction uses the requested positions of the generated horizon."""
    y = pd.Series(np.arange(20, dtype=float), index=pd.RangeIndex(10, 30))
    forecaster, model = _fit_fake(monkeypatch, y)

    prediction = forecaster.predict(fh=[2, 5])

    assert prediction.index.equals(pd.Index([31, 34]))
    np.testing.assert_array_equal(prediction.to_numpy(), [4901, 4904])
    assert model.horizons == [5]


def test_tabby_quantile_mapping_and_interpolation(monkeypatch):
    """Native quantiles are selected and intermediate levels interpolated."""
    forecaster, _ = _fit_fake(monkeypatch, pd.Series(np.arange(12, dtype=float)))

    prediction = forecaster.predict_quantiles(
        fh=[3], alpha=[0.1, 0.100000005, 0.105, 0.99]
    )

    assert prediction.columns.equals(
        pd.MultiIndex.from_product([[0], [0.1, 0.100000005, 0.105, 0.99]])
    )
    np.testing.assert_allclose(prediction.to_numpy(), [[902, 902, 952, 9802]])


def test_tabby_quantiles_reject_out_of_range_alpha(monkeypatch):
    """Quantiles outside Tabby's native support are rejected."""
    forecaster, _ = _fit_fake(monkeypatch, pd.Series(np.arange(12, dtype=float)))

    with pytest.raises(ValueError, match=r"\[0.01, 0.99\]"):
        forecaster.predict_quantiles(fh=[1], alpha=[0.009])


def test_tabby_predict_interval_uses_quantiles(monkeypatch):
    """The base interval conversion uses Tabby's quantile implementation."""
    forecaster, _ = _fit_fake(monkeypatch, pd.Series(np.arange(12, dtype=float)))

    prediction = forecaster.predict_interval(fh=[1], coverage=0.9)

    assert prediction.shape == (1, 2)
    np.testing.assert_allclose(prediction.to_numpy(), [[400, 9400]])


def test_tabby_truncates_context_and_preserves_nan(monkeypatch):
    """The model receives the recent context only, including missing values."""
    values = np.arange(10_000, dtype=float)
    values[-10] = np.nan
    forecaster, model = _fit_fake(monkeypatch, pd.Series(values))

    forecaster.predict(fh=[1])

    context = model.contexts[0][0]
    assert len(context) == 8192 - 128
    np.testing.assert_array_equal(context, values[-8064:])


def test_tabby_vectorizes_multivariate_series(monkeypatch):
    """Multivariate targets are forecast one variable at a time."""
    y = pd.DataFrame(
        {"left": np.arange(12, dtype=float), "right": np.arange(12, dtype=float)}
    )
    forecaster, model = _fit_fake(monkeypatch, y)

    prediction = forecaster.predict(fh=[1, 2])

    assert list(prediction.columns) == ["left", "right"]
    assert len(model.horizons) == 2
    np.testing.assert_array_equal(prediction.to_numpy(), [[4900, 4900], [4901, 4901]])


def test_tabby_model_capacity_is_enforced(monkeypatch):
    """The forecast horizon cannot exceed the checkpoint's patch capacity."""
    forecaster, model = _fit_fake(monkeypatch, pd.Series(np.arange(12, dtype=float)))

    with pytest.raises(ValueError, match="maximum supported step=8176"):
        forecaster.predict(fh=[8177])
    assert model.horizons == []


def test_tabby_requires_remote_code_opt_in():
    """Loading the checkpoint requires explicit trust in its remote code."""
    forecaster = TabbyForecaster()

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        forecaster._load_model()


def test_tabby_model_cache(monkeypatch):
    """Identical model settings reuse the cached loaded model."""
    if not _check_soft_dependencies("transformers", severity="none"):
        pytest.skip("transformers is not installed")
    import torch
    from transformers import AutoModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FakeLoadedModel:
        def __init__(self):
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    loaded_model = FakeLoadedModel()
    calls = []

    def fake_from_pretrained(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return loaded_model

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    model_path = "sktime-test/tabby-cache"
    first = TabbyForecaster(
        model_path=model_path, revision="test-revision", trust_remote_code=True
    )
    second = TabbyForecaster(
        model_path=model_path, revision="test-revision", trust_remote_code=True
    )

    assert first._load_model() is second._load_model()
    assert len(calls) == 1
    assert calls[0][1]["trust_remote_code"] is True
    assert calls[0][1]["revision"] == "test-revision"
    assert loaded_model.device == "cpu"
    assert loaded_model.evaluated
