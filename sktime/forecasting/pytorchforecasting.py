# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""
import functools
from typing import Any, Dict, List, Optional

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _PytorchForecastingAdapter,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

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
        "capability:global_forecasting": True,
        "requires_X": True,
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
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
        )
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names

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
        return (
            {}
            if self.allowed_encoder_known_variable_names is None
            else {
                "allowed_encoder_known_variable_names\
                    ": self.allowed_encoder_known_variable_names,
            }
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    }
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        "optimizer": "Adam",
                        "log_val_interval": 1,
                    },
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping
            from pytorch_forecasting.metrics import QuantileLoss

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    }
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        "loss": QuantileLoss(),
                        "optimizer": "Adam",
                        "log_val_interval": 1,
                    },
                },
            ]

        return params


class PytorchForecastingNBeats(_PytorchForecastingAdapter):
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
        "capability:global_forecasting": True,
        "ignores-exogeneous-X": True,
    }

    def __init__(
        self: "PytorchForecastingNBeats",
        model_params: Optional[Dict[str, Any]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
        )

    @functools.cached_property
    def algorithm_class(self: "PytorchForecastingNBeats"):
        """Import underlying pytorch-forecasting algorithm class."""
        from pytorch_forecasting import NBeats

        return NBeats

    @functools.cached_property
    def algorithm_parameters(self: "PytorchForecastingNBeats") -> dict:
        """Get keyword parameters for the NBeats class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("pytorch_forecasting", severity="error")
        except ModuleNotFoundError:
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    }
                },
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                    }
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                },
            ]

        return params
