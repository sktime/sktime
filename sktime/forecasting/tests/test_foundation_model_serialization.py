"""Regression tests for foundation model forecaster serialization."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pickle

import pytest

from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.chronos2 import Chronos2Forecaster


def _make_chronos(monkeypatch):
    """Create a Chronos forecaster without loading model metadata."""

    def _initialize_model_type(self):
        self._default_config = self._default_chronos_config.copy()
        self._config = self._default_config.copy()

    monkeypatch.setattr(
        ChronosForecaster,
        "_initialize_model_type",
        _initialize_model_type,
    )

    return ChronosForecaster(
        model_path="amazon/chronos-t5-tiny",
        ignore_deps=True,
    )


SERIALIZATION_CASES = [
    (
        "chronos",
        _make_chronos,
        "model_pipeline",
        "_load_pipeline",
    ),
    (
        "chronos2",
        lambda monkeypatch: Chronos2Forecaster(ignore_deps=True),
        "model_pipeline",
        "_load_pipeline",
    ),
]


@pytest.mark.parametrize(
    ("_case_name", "factory", "cached_field", "loader_name"),
    SERIALIZATION_CASES,
    ids=[case[0] for case in SERIALIZATION_CASES],
)
def test_cached_foundation_models_are_dropped_and_reloaded(
    monkeypatch, _case_name, factory, cached_field, loader_name
):
    """Cached zero-shot model handles should not block pickling."""
    est = factory(monkeypatch)
    est._is_fitted = True
    setattr(est, cached_field, lambda: None)

    restored_model = object()
    loader_calls = {"count": 0}

    def _loader(self):
        loader_calls["count"] += 1
        return restored_model

    monkeypatch.setattr(type(est), loader_name, _loader)

    loaded = pickle.loads(pickle.dumps(est))

    assert getattr(loaded, cached_field) is None

    loaded._ensure_cached_models_loaded()

    assert getattr(loaded, cached_field) is restored_model
    assert loader_calls["count"] == 1

    loaded._ensure_cached_models_loaded()

    assert loader_calls["count"] == 1


@pytest.mark.parametrize(
    ("_case_name", "factory", "cached_field", "loader_name"),
    SERIALIZATION_CASES,
    ids=[case[0] for case in SERIALIZATION_CASES],
)
def test_cached_foundation_models_are_not_reloaded_when_unfitted(
    monkeypatch, _case_name, factory, cached_field, loader_name
):
    """Unfitted estimators should stay without cached model handles after pickle."""
    est = factory(monkeypatch)
    est._is_fitted = False
    setattr(est, cached_field, lambda: None)

    loader_calls = {"count": 0}

    def _loader(self):
        loader_calls["count"] += 1
        return object()

    monkeypatch.setattr(type(est), loader_name, _loader)

    loaded = pickle.loads(pickle.dumps(est))
    loaded._ensure_cached_models_loaded()

    assert getattr(loaded, cached_field) is None
    assert loader_calls["count"] == 0
