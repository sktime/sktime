"""Tests for TinyTimeMixerForecaster."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_basic_functionality():
    """Test basic forecaster functionality without few-shot learning."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 8, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict()

    # Basic assertions
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_exogenous_variables():
    """Test forecaster with exogenous variables."""
    from sktime.datasets import load_longley
    from sktime.forecasting.ttm import TinyTimeMixerForecaster
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, _, X_train, X_future = temporal_train_test_split(y, X, test_size=2)

    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 6, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y_train, X=X_train, fh=[1, 2])
    y_pred = forecaster.predict(X=X_future)

    # Basic assertions
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_backward_compatibility():
    """Test that existing code works without few-shot parameters."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    # Test original usage pattern (no few-shot parameters)
    forecaster = TinyTimeMixerForecaster(
        model_path=None,
        fit_strategy="full",
        config={"context_length": 8, "prediction_length": 2},
        training_args={
            "max_steps": 2,
            "output_dir": "test_output",
            "per_device_train_batch_size": 4,
            "report_to": "none",
        },
    )

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict()

    # Verify backward compatibility
    assert y_pred is not None
    assert len(y_pred) == 2
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_few_shot_capability():
    """Test few-shot learning functionality with different ratios."""
    from sktime.datasets import load_airline
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    y = load_airline()

    # Test different few-shot ratios
    ratios = [0.2, 0.5, 0.8]

    for ratio in ratios:
        forecaster = TinyTimeMixerForecaster(
            model_path=None,
            fit_strategy="full",
            few_shot_ratio=ratio,
            few_shot_random_state=42,
            config={"context_length": 8, "prediction_length": 2},
            training_args={
                "max_steps": 2,
                "output_dir": "test_output",
                "per_device_train_batch_size": 4,
                "report_to": "none",
            },
        )

        forecaster.fit(y, fh=[1, 2])
        y_pred = forecaster.predict()

        # Verify few-shot functionality
        assert y_pred is not None
        assert len(y_pred) == 2
        assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TinyTimeMixerForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ttm_estimator_check():
    """Run standard estimator checks for TTM."""
    from sktime.forecasting.ttm import TinyTimeMixerForecaster
    from sktime.utils import check_estimator

    check_estimator(TinyTimeMixerForecaster, raise_exceptions=True)
