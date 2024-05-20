# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for HFTransformersForecaster."""
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datasets import load_longley
from sktime.forecasting.hf_transformers_forecaster import HFTransformersForecaster
from sktime.tests.test_switch import run_test_for_class

__author__ = ["geetu040"]

y, X = load_longley()


@pytest.mark.parametrize(
    "fit_strategy",
    [
        "minimal",
        "full",
    ],
)
@pytest.mark.skipif(
    not run_test_for_class([HFTransformersForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fit_strategy(fit_strategy) -> None:
    """Test with single endogenous without exogenous."""
    # initialize forecaster
    forecaster = HFTransformersForecaster(
        model_path="huggingface/informer-tourism-monthly",
        fit_strategy=fit_strategy,
        training_args={
            "num_train_epochs": 1,
            "output_dir": "test_output",
            "per_device_train_batch_size": 32,
        },
        config={
            "lags_sequence": [1, 2, 3],
            "context_length": 2,
            "prediction_length": 4,
            "use_cpu": True,
            "label_length": 2,
        },
    )
    # fit the forecaster
    forecaster.fit(y=y)
    # make predictions
    forecaster.predict([1, 2, 3])


@pytest.mark.parametrize(
    "fit_strategy",
    [
        "lora",
        "loha",
        "adalora",
    ],
)
@pytest.mark.skipif(
    not run_test_for_class([HFTransformersForecaster]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies("peft", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_peft_fit_strategy(fit_strategy) -> None:
    """Test with single endogenous without exogenous."""
    # initialize forecaster
    forecaster = HFTransformersForecaster(
        model_path="huggingface/autoformer-tourism-monthly",
        fit_strategy=fit_strategy,
        training_args={
            "num_train_epochs": 1,
            "output_dir": "test_output",
            "per_device_train_batch_size": 32,
        },
        config={
            "lags_sequence": [1, 2, 3],
            "context_length": 2,
            "prediction_length": 4,
            "use_cpu": True,
            "label_length": 2,
        },
    )
    # fit the forecaster
    forecaster.fit(y=y)
    # make predictions
    forecaster.predict([1, 2, 3])
