# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from pytorch-forecasting."""

import functools
from typing import Any, Optional

from sktime.forecasting.base.adapters._pytorchforecasting import (
    _PytorchForecastingAdapter,
)
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["XinyuWu"]


class PytorchForecastingTFT(_PytorchForecastingAdapter):
    """pytorch-forecasting Temporal Fusion Transformer model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting TFT model [1]_
        for example: {"lstm_layers": 3, "hidden_continuous_size": 10}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingTFT",
        model_params: Optional[dict[str, Any]] = None,
        allowed_encoder_known_variable_names: Optional[list[str]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
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
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_val_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]
        else:
            from lightning.pytorch.callbacks import EarlyStopping

            # from pytorch_forecasting.metrics import QuantileLoss

            early_stop_callback = EarlyStopping(
                monitor="train_loss",
                min_delta=1e-2,
                patience=3,
                verbose=False,
                mode="min",
            )
            params = [
                {
                    "trainer_params": {
                        "max_epochs": 1,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "hidden_size": 10,
                        "dropout": 0.1,
                        # "loss": QuantileLoss(),
                        # can not pass test_set_params and test_set_params_sklearn
                        # QuantileLoss() != QuantileLoss()
                        "optimizer": "Adam",
                        # avoid jdb78/pytorch-forecasting#1571 bug in the CI
                        "log_val_interval": -1,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params


class PytorchForecastingNBeats(_PytorchForecastingAdapter):
    """pytorch-forecasting NBeats model.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the pytorch-forecasting NBeats model [1]_
        for example: {"num_blocks": [5, 5], "widths": [128, 1024]}
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [2]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be infered from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.nbeats.NBeats.html
    .. [2] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        # "authors": ["XinyuWu"],
        # "maintainers": ["XinyuWu"],
        # "python_dependencies": "pytorch_forecasting"
        # inherited from _PytorchForecastingAdapter
        # estimator type
        # --------------
        "python_dependencies": ["pytorch_forecasting>=1.0.0", "torch", "lightning"],
        "capability:global_forecasting": True,
        "ignores-exogeneous-X": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "scitype:y": "univariate",
    }

    def __init__(
        self: "PytorchForecastingNBeats",
        model_params: Optional[dict[str, Any]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
    ) -> None:
        super().__init__(
            model_params,
            dataset_params,
            train_to_dataloader_params,
            validation_to_dataloader_params,
            trainer_params,
            model_path,
            random_log_path,
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
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
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
                        "limit_train_batches": 10,  # for quick test
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
                {
                    "trainer_params": {
                        "callbacks": [early_stop_callback],
                        "max_epochs": 3,  # for quick test
                        "limit_train_batches": 10,  # for quick test
                    },
                    "model_params": {
                        "num_blocks": [5, 5],
                        "num_block_layers": [5, 5],
                        "log_interval": 10,
                        "backcast_loss_ratio": 1.0,
                    },
                    "dataset_params": {
                        "max_encoder_length": 3,
                    },
                    "random_log_path": True,  # fix multiprocess file access error in CI
                },
            ]

        return params
