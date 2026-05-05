"""Tests for Chronos zero-shot weight loading migration.

Tests that ChronosForecaster loads model weights at construction time
(in __post_init__) rather than in _fit(), ensuring:
1. No redundant weight reloading on repeated fit() calls
2. The pretrain -> fit -> predict workflow functions correctly
3. Pipeline persistence across serialization/deserialization
"""

__author__ = ["Keykyrios"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.dependencies import _check_estimator_deps


@pytest.fixture
def _check_chronos_deps():
    """Skip test if Chronos dependencies are not available."""
    from sktime.forecasting.chronos import ChronosForecaster

    try:
        _check_estimator_deps(ChronosForecaster, severity="error")
    except ModuleNotFoundError:
        pytest.skip("Chronos soft dependencies not available")


@pytest.fixture
def forecaster(_check_chronos_deps):
    """Create a ChronosForecaster instance."""
    from sktime.forecasting.chronos import ChronosForecaster

    return ChronosForecaster("amazon/chronos-t5-tiny")


@pytest.fixture
def airline_data():
    """Load airline dataset split into train/test."""
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split

    y = load_airline()
    return temporal_train_test_split(y)


@pytest.fixture
def panel_data():
    """Create panel data for pretrain testing."""
    y_panel = pd.DataFrame(
        {"y": np.random.default_rng(42).standard_normal(60)},
        index=pd.MultiIndex.from_arrays(
            [
                np.repeat(["ts1", "ts2", "ts3"], 20),
                np.tile(pd.date_range("2000", periods=20, freq="ME"), 3),
            ],
            names=["instance", "time"],
        ),
    )
    return y_panel


def test_pipeline_loaded_at_construction(forecaster):
    """Test that model pipeline is loaded during __post_init__."""
    assert hasattr(forecaster, "model_pipeline")
    assert forecaster.model_pipeline is not None


def test_fit_does_not_reload_pipeline(forecaster, airline_data):
    """Test that fit() reuses the pipeline loaded at construction."""
    y_train, _ = airline_data
    pipeline_before = id(forecaster.model_pipeline)

    forecaster.fit(y_train)

    pipeline_after = id(forecaster.model_pipeline)
    assert pipeline_before == pipeline_after, (
        "Pipeline was reloaded during fit(). "
        "Zero-shot models should load weights once at construction."
    )


def test_repeated_fit_no_reload(forecaster, airline_data):
    """Test that repeated fit() calls don't reload the pipeline."""
    y_train, _ = airline_data
    pipeline_id_init = id(forecaster.model_pipeline)

    forecaster.fit(y_train)
    pipeline_id_fit1 = id(forecaster.model_pipeline)

    forecaster.fit(y_train)
    pipeline_id_fit2 = id(forecaster.model_pipeline)

    assert pipeline_id_init == pipeline_id_fit1 == pipeline_id_fit2


def test_pretrain_capability_tag(forecaster):
    """Test that capability:pretrain tag is set to True."""
    assert forecaster.get_tag("capability:pretrain") is True


def test_pretrain_fit_predict_flow(forecaster, airline_data, panel_data):
    """Test the full pretrain -> fit -> predict lifecycle."""
    y_train, y_test = airline_data

    # pretrain on panel data
    forecaster.pretrain(panel_data)
    assert forecaster.state == "pretrained"
    pipeline_after_pretrain = id(forecaster.model_pipeline)

    # fit on single series (should not reload pipeline)
    forecaster.fit(y_train)
    assert forecaster.state == "fitted"
    pipeline_after_fit = id(forecaster.model_pipeline)
    assert pipeline_after_pretrain == pipeline_after_fit

    # predict
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    assert len(y_pred) == len(y_test)


def test_fit_predict_without_pretrain(forecaster, airline_data):
    """Test that standard fit -> predict still works without pretrain."""
    y_train, y_test = airline_data

    forecaster.fit(y_train)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)

    assert len(y_pred) == len(y_test)
    assert forecaster.state == "fitted"


def test_pickle_roundtrip(forecaster, airline_data):
    """Test that pickling and unpickling restores the pipeline."""
    import pickle

    y_train, y_test = airline_data
    forecaster.fit(y_train)

    # Pickle and unpickle
    pickled = pickle.dumps(forecaster)
    restored = pickle.loads(pickled)

    # Pipeline should be lazily reloaded
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = restored.predict(fh)
    assert len(y_pred) == len(y_test)
