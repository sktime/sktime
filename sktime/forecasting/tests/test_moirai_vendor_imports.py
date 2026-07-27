"""Tests for MOIRAI vendored import handling."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize(
    [
        "forecaster_path",
        "class_name",
        "forecast_path",
        "forecast_name",
        "module_path",
        "module_name",
    ],
    [
        (
            "sktime.forecasting.moirai",
            "MOIRAIForecaster",
            "sktime.libs.uni2ts.forecast",
            "MoiraiForecast",
            "sktime.libs.uni2ts.moirai_module",
            "MoiraiModule",
        ),
        (
            "sktime.forecasting.moirai2",
            "Moirai2Forecaster",
            "sktime.libs.uni2ts.moirai2_forecast",
            "Moirai2Forecast",
            "sktime.libs.uni2ts.moirai2_module",
            "Moirai2Module",
        ),
    ],
)
def test_vendor_loader_does_not_alias_uni2ts(
    monkeypatch,
    forecaster_path,
    class_name,
    forecast_path,
    forecast_name,
    module_path,
    module_name,
):
    """Vendor model loading should not install a top-level uni2ts alias."""
    from importlib import import_module

    forecaster_class = getattr(import_module(forecaster_path), class_name)
    loaded_module = object()

    class DummyModule:
        @classmethod
        def from_pretrained(cls, checkpoint_path):
            assert checkpoint_path.startswith("Salesforce/")
            return loaded_module

    class DummyForecast:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    forecast_module = ModuleType(forecast_path)
    setattr(forecast_module, forecast_name, DummyForecast)
    model_module = ModuleType(module_path)
    setattr(model_module, module_name, DummyModule)

    monkeypatch.setitem(sys.modules, forecast_path, forecast_module)
    monkeypatch.setitem(sys.modules, module_path, model_module)
    monkeypatch.delitem(sys.modules, "uni2ts", raising=False)

    forecaster = forecaster_class("Salesforce/moirai-test")
    model = forecaster._instantiate_vendor_model({"prediction_length": 1})

    assert model.kwargs["module"] is loaded_module
    assert "uni2ts" not in sys.modules


def test_upstream_config_targets_use_vendor_namespace():
    """Upstream Hydra targets should be mapped without an import alias."""
    from sktime.libs.uni2ts.moirai_module import _use_vendor_module_paths

    config = {
        "_target_": "uni2ts.distribution.mixture.MixtureOutput",
        "components": [
            {"_target_": "uni2ts.distribution.normal.NormalOutput"},
            {"_target_": "external.package.Output"},
        ],
    }

    mapped = _use_vendor_module_paths(config)

    assert mapped == {
        "_target_": "sktime.libs.uni2ts.distribution.mixture.MixtureOutput",
        "components": [
            {"_target_": "sktime.libs.uni2ts.distribution.normal.NormalOutput"},
            {"_target_": "external.package.Output"},
        ],
    }
    assert config["_target_"] == "uni2ts.distribution.mixture.MixtureOutput"


def test_checkpoint_safe_globals_are_scoped(monkeypatch):
    """Legacy checkpoint globals should be removed after model loading."""
    import sktime.forecasting.moirai as moirai

    events = []
    existing_alias = (object(), "uni2ts.example.Existing")
    new_alias = (object(), "uni2ts.example.New")
    aliases = [existing_alias, new_alias]

    @contextmanager
    def safe_globals(values):
        events.append(("enter", values))
        yield
        events.append(("exit", values))

    torch_module = ModuleType("torch")
    torch_module.serialization = SimpleNamespace(
        get_safe_globals=lambda: [existing_alias],
        safe_globals=safe_globals,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(
        moirai,
        "_get_moirai_checkpoint_safe_globals",
        lambda: aliases,
    )

    class DummyModel:
        @classmethod
        def load_from_checkpoint(cls, **kwargs):
            events.append(("load", kwargs))
            return "model"

    result = moirai._load_moirai_checkpoint(DummyModel, {"checkpoint_path": "model"})

    assert result == "model"
    assert events == [
        ("enter", [new_alias]),
        ("load", {"checkpoint_path": "model"}),
        ("exit", [new_alias]),
    ]
