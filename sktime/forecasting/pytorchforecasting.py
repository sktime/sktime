# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""
import functools
from typing import Any, Dict, List, Optional

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _PytorchForecastingAdapter,
)

__author__ = ["XinyuWu"]


class PytorchForecastingTFT(_PytorchForecastingAdapter):
    """pytorch-forecasting Temporal Fusion Transformer model."""

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting", "torch", "lightning"],
    }

    def __init__(
        self: "PytorchForecastingTFT",
        model_params: Optional[Dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_params,
            allowed_encoder_known_variable_names,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingTFT"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import TemporalFusionTransformer

        return TemporalFusionTransformer

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingTFT") -> dict:
        """Get keyword parameters for the TFT class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return {}
