# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TinyTimeMixerForecaster."""

__author__ = ["geetu040"]

import pandas as pd
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster, return_reason=True),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ttm_exogenous_variables_basic():
    """Test basic exogenous variable support in TTM."""
    # Load test data with exogenous variables
    y, X = load_longley()

    # Create forecaster with full training
    forecaster = TinyTimeMixerForecaster(
        model_path=None,  # type: ignore
        fit_strategy="full",
        config={
            "context_length": 8,
            "prediction_length": 2,
            "num_input_channels": 1,
        },
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    # Test fitting with exogenous variables
    forecaster.fit(y, X=X, fh=[1, 2])

    # Test prediction with exogenous variables
    future_X = X.iloc[-2:].copy()
    # Create future index with the same frequency as the original data
    # For PeriodIndex, we need to handle frequency differently
    if isinstance(y.index, pd.PeriodIndex):
        # For PeriodIndex, create future periods
        future_X.index = pd.PeriodIndex(
            [y.index[-1] + 1, y.index[-1] + 2], freq=y.index.freq
        )
    else:
        # For other index types, use date_range
        future_X.index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(days=1), periods=2, freq="Y"
        )

    predictions = forecaster.predict(X=future_X)
    print("")
    assert predictions is not None


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster, return_reason=True),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ttm_exogenous_variables_validation_split():
    """Test that validation split works with exogenous variables."""
    y, X = load_longley()

    forecaster = TinyTimeMixerForecaster(
        model_path=None,  # type: ignore
        fit_strategy="full",
        validation_split=0.2,
        config={
            "context_length": 8,
            "prediction_length": 2,
            "num_input_channels": 1,
        },
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    # Should work with validation split and exogenous variables
    forecaster.fit(y, X=X, fh=[1, 2])
    predictions = forecaster.predict()
    assert predictions is not None


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster, return_reason=True),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ttm_exogenous_variables_backward_compatibility():
    """Test that existing functionality works without exogenous variables."""
    y, _ = load_longley()

    forecaster = TinyTimeMixerForecaster(
        model_path=None,  # type: ignore
        fit_strategy="full",
        config={
            "context_length": 8,
            "prediction_length": 2,
            "num_input_channels": 1,
        },
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    # Should work without exogenous variables (backward compatibility)
    forecaster.fit(y, fh=[1, 2])
    predictions = forecaster.predict()
    assert predictions is not None
