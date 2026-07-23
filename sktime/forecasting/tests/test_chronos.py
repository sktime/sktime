"""Tests for ChronosForecaster."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mahesh-sadupalli"]

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sktime.forecasting.chronos import ChronosForecaster, ChronosBoltStrategy
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting"),
    reason="run test only incrementally (if requested)",
)
def test_chronos_predict_without_context_length_on_config():
    """Regression test for #10592.

    When use_source_package=True the external chronos package (v2+) returns a
    plain T5Config that does not carry context_length.  predict() must not raise
    AttributeError and should fall back gracefully (skip explicit truncation).
    """
    pytest.importorskip("torch")
    import torch

    # Build a mock pipeline whose model.config has NO context_length,
    # mimicking the external chronos v2.3.1 T5Config behaviour.
    mock_config = MagicMock(spec=[])  # spec=[] → no attributes exposed
    mock_model = MagicMock()
    mock_model.config = mock_config

    mock_pipeline = MagicMock()
    mock_pipeline.model = mock_model
    del mock_pipeline.context_length  # also absent at pipeline level

    prediction_length = 3
    mock_values = np.arange(1, prediction_length + 1, dtype=float)

    bolt_config = {
        "limit_prediction_length": False,
        "torch_dtype": torch.bfloat16,
        "device_map": "cpu",
    }
    mock_strategy = MagicMock(spec=ChronosBoltStrategy)
    mock_strategy.initialize_config.return_value = bolt_config.copy()
    mock_strategy.predict.return_value = mock_values

    n = 24
    index = pd.period_range("2020-01", periods=n, freq="M")
    y = pd.Series(range(n), index=index, dtype=float)

    # Patch _initialize_model_type (prevents HuggingFace Hub network call in
    # __post_init__) and _load_pipeline (prevents chronos package import).
    with (
        patch.object(
            ChronosForecaster,
            "_initialize_model_type",
            lambda self: _setup_strategy(self, mock_strategy, bolt_config),
        ),
        patch.object(
            ChronosForecaster,
            "_load_pipeline",
            return_value=mock_pipeline,
        ),
    ):
        forecaster = ChronosForecaster(
            model_path="amazon/chronos-bolt-base",
            ignore_deps=True,
            use_source_package=True,
        )
        # full fit() sets all base-class state (_is_vectorized, cutoff, etc.)
        forecaster.fit(y, fh=[1, 2, 3])

    forecaster.model_pipeline = mock_pipeline

    # Should not raise AttributeError on missing context_length
    preds = forecaster.predict(fh=[1, 2, 3])
    assert len(preds) == prediction_length


def _setup_strategy(forecaster, mock_strategy, bolt_config):
    """Helper to wire up the mock strategy as _initialize_model_type would."""
    forecaster.model_strategy = mock_strategy
    forecaster._default_config = bolt_config.copy()
    forecaster._config = bolt_config.copy()
