"""Implements TimeMOE forecaster."""

__all__ = ["TimeMoEForecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster, _GlobalForecastingDeprecationMixin
from sktime.utils.singleton import _multiton


class TimeMoEForecaster(_GlobalForecastingDeprecationMixin, BaseForecaster):
    """
    Interface for TimeMOE forecaster.

    TimeMoE is a decoder-only time series foundational model that uses a mixture
    of experts algorithm to make predictions. designed to operate in an auto-regressive
    manner, enabling universal forecasting with arbitrary prediction horizons
    and context lengths of up to 4096. This method has been proposed in [2]_ and the
    official code is available at [2]_.

    Supports:

    - zero-shot forecasting via ``fit`` + ``predict``
    - fine-tuning of a pretrained checkpoint via ``pretrain``
    - training from scratch via ``pretrain`` with ``model_path=None``

    Parameters
    ----------
    model_path: str or None, default="Maple728/TimeMoE-50M"
        Path to the TimeMOE model. This can be:

        - A model ID from the HuggingFace Hub, e.g., "Maple728/TimeMoE-50M"
        - A local directory containing the model files, specified as an absolute or
          relative path to the current working directory
          The path should point to a directory containing the model weights and
          configuration files in the format expected by the HuggingFace Transformers
          library.
        - ``None`` to initialize from ``config`` with random weights (from-scratch)

    config: dict, optional
        A dictionary specifying the configuration of the TimeMOE model.
        The available configuration options include hyperparameters that control
        the prediction behavior, sampling, and hardware utilization.

        - input_size: int, default=1
            The size of the input time series.
        - hidden_size: int, default=4096
            The size of the hidden layers in the TimeMOE model.
        - intermediate_size: int, default=22016
            The size of the intermediate layers in the TimeMOE model.
        - horizon_lengths: list[int], default=[1]
            The prediction horizon length.
        - num_hidden_layers: int, default=32
            The number of hidden layers in the TimeMOE model.
        - num_attention_heads: int, default=32
            The number of attention heads in the TimeMOE model.
        - num_experts_per_tok: int, default=2
            The number of experts per token in the TimeMOE model.
        - num_experts: int, default=1
            The number of experts in the TimeMOE model.
        - max_position_embeddings: int, default=32768
            The maximum position embeddings in the TimeMOE model.
        - rms_norm_eps: float, default=1e-6
            The epsilon value for RMS normalization in the TimeMOE model.
        - rope_theta: int, default=10000
            Initialise theta for RoPE (Rotational Positional Embeddings).
        - attention_dropout: float, default=0.1
            The dropout rate for attention layers in the TimeMOE model.
        - apply_aux_loss: bool, default=True
            Whether to apply auxiliary loss in the TimeMOE model.
        - router_aux_loss_factor: float, default=0.02
            The auxiliary loss factor for the router in the TimeMOE model.
        - tie_word_embeddings: bool, default=False
            Whether to tie word embeddings in the TimeMOE model.

        Architecture keys are used when ``model_path=None`` (from-scratch) and
        ignored otherwise.

    seed: int, optional (default=None)
        Seed for reproducibility.

    use_source_package: bool, optional (default=False)
        If True, the model will be loaded directly from the source package ``TimeMoE``.
        This is useful if you want to bypass the local version of the package
        or when working in an environment where the latest updates from the source
        package are needed. If False, the model will be loaded from the local version
        of package maintained in sktime. To install the source package,
        follow the instructions here [1]_.

    ignore_deps: bool, optional, default=False
        If True, dependency checks will be ignored, and the user is expected to handle
        the installation of required packages manually. If False, the class will enforce
        the default dependencies required for Chronos.

    context_length : int, optional (default=None)
        Sliding-window length for ``pretrain``. Defaults to ``1024`` when ``None``.
        For small datasets, use a shorter length with ``stride=1``.

    stride : int, optional (default=None)
        Sliding-window stride for ``pretrain``. Defaults to ``context_length``.

    training_args : dict, optional (default=None)
        Keyword arguments used for training.
        Supports all arguments by ``transformers.TrainingArguments`` availble at
        https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.

        Additionally, the following arguments are supported:
        - min_learning_rate: float, default=0
            Minimum learning rate for cosine_schedule

    References
    ----------
    .. [1] https://github.com/Time-MoE/Time-MoE
    .. [2] Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye and others
    Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts

    Examples
    --------
    Zero-shot forecasting:

    >>> from sktime.forecasting.timemoe import TimeMoEForecaster
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=5)
    >>> forecaster = TimeMoEForecaster("Maple728/TimeMoE-50M")
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3], y = y_test)  # doctest: +SKIP

    Fine-tuning of a pretrained checkpoint:

    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y_panel = _make_hierarchical(  # doctest: +SKIP
    ...     hierarchy_levels=(3,), min_timepoints=64, max_timepoints=128,
    ... )
    >>> forecaster = TimeMoEForecaster(  # doctest: +SKIP
    ...     model_path="Maple728/TimeMoE-50M",
    ...     context_length=32,
    ...     stride=1,
    ...     training_args={"max_steps": 10, "per_device_train_batch_size": 2},
    ... )
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> forecaster.fit(load_airline())  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Training from scratch:

    >>> forecaster = TimeMoEForecaster(  # doctest: +SKIP
    ...     model_path=None,
    ...     config={
    ...         "hidden_size": 64,
    ...         "intermediate_size": 128,
    ...         "num_hidden_layers": 2,
    ...         "num_attention_heads": 4,
    ...         "num_experts": 2,
    ...         "num_experts_per_tok": 1,
    ...         "horizon_lengths": [1],
    ...         "max_position_embeddings": 128,
    ...     },
    ...     context_length=32,
    ...     stride=1,
    ...     training_args={"max_steps": 10},
    ... )
    >>> forecaster.pretrain(y_panel)  # doctest: +SKIP
    >>> forecaster.fit(load_airline())  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Maple728", "KimMeen", "PranavBhatP", "Faakhir30"],
        # abdulfatir and lostella for amazon-science/chronos-forecasting
        "maintainers": ["PranavBhatP", "Faakhir30"],
        "python_dependencies": ["torch", "transformers<=4.40.1", "accelerate<=0.28.0"],
        # estimator type
        # --------------
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:pretrain": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.timemoe"],
    }

    def __init__(
        self,
        model_path: str | None = "Maple728/TimeMoE-50M",
        config: dict = None,
        seed: int = None,
        use_source_package: bool = False,
        ignore_deps: bool = False,
        context_length: int = None,
        stride: int = None,
        training_args: dict = None,
    ):
        self.seed = seed
        self.config = config
        self.model_path = model_path
        self.use_source_package = use_source_package
        self.ignore_deps = ignore_deps
        self.context_length = context_length
        self.stride = stride
        self.training_args = training_args

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values condition on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])
        elif self.use_source_package:
            self.set_tags(python_dependencies=["timemoe"])
        else:
            self.set_tags(
                python_dependencies=[
                    "torch",
                    "transformers<=4.40.1",
                    "accelerate<=0.28.0",
                ]
            )

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        self._seed = np.random.randint(0, 2**31) if self.seed is None else self.seed

        _config = self._get_default_config()
        _config.update(self.config if self.config is not None else {})
        self._config = _config

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain / fine-tune using upstream Time-MoE Trainer + window dataset."""
        from sktime.libs.timemoe.hf_trainer import (
            TimeMoeTrainer,
            TimeMoETrainingArguments,
        )
        from sktime.libs.timemoe.time_moe_window_dataset import TimeMoEWindowDataset
        from sktime.libs.timemoe.ts_dataset import SeriesListDataset

        self.model_ = self._load_model()

        context_length = (
            self.context_length if self.context_length is not None else 1024
        )
        train_ds = TimeMoEWindowDataset(
            SeriesListDataset(_prepare_series_list(y)),
            context_length=context_length,
            prediction_length=0,
            stride=self.stride,
        )

        training_args = (
            deepcopy(self.training_args) if self.training_args is not None else {}
        )
        training_args.setdefault("output_dir", "tmp_timemoe_trainer")
        training_args.setdefault("report_to", [])
        training_args = TimeMoETrainingArguments(**training_args)

        trainer = TimeMoeTrainer(
            model=self.model_,
            args=training_args,
            train_dataset=train_ds,
        )
        trainer.train()

        self.model_ = trainer.model
        return self

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        config = self._config
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            config["input_size"] = y.shape[1]
        else:
            config["input_size"] = 1
        self._config = config
        self.model_ = self._load_model()

        return self

    def _load_model(self):
        """Load model, reusing ``model_`` after pretrain."""
        if hasattr(self, "model_") and self.model_ is not None:
            return self.model_

        model = _CachedTimeMoE(
            key=self._get_unique_timemoe_key(),
            timemoe_kwargs=self._get_timemoe_kwargs(),
            use_source_package=self.use_source_package,
            config=self._config,
        ).load_from_checkpoint()
        return model

    def _get_timemoe_kwargs(self):
        """Get the kwargs for TimeMoE model."""
        kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

        return kwargs

    def _get_unique_timemoe_key(self):
        """Get a unique key for TimeMoE model."""
        kwargs_plus_model_path = {
            **self._get_timemoe_kwargs(),
            "use_source_package": self.use_source_package,
            "config": self._config,
        }

        return str(sorted(kwargs_plus_model_path.items()))

    def _get_default_config(self):
        """Return the default configuration for TimeMoE model.

        Returns
        -------
        dict
            The default configuration for TimeMoE model.
        """
        import torch

        default_config = {
            "input_size": 1,
            "hidden_size": 4096,
            "intermediate_size": 22016,
            "horizon_lengths": [1],
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": None,
            "hidden_act": "silu",
            "num_experts_per_tok": 2,
            "num_experts": 1,
            "max_position_embeddings": 32768,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "use_dense": False,
            "rope_theta": 10000,
            "attention_dropout": 0.0,
            "apply_aux_loss": True,
            "router_aux_loss_factor": 0.02,
            "tie_word_embeddings": False,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
        }
        return default_config

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        y : pd.Series, optional (default=None)
            Optional series to use instead of the series passed in fit.

        Returns
        -------
        y_pred : pd.DataFrame
            Predicted forecasts.
        """
        import torch
        import transformers

        transformers.set_seed(self._seed)
        if fh is not None:
            prediction_length = int(max(fh.to_relative(self.cutoff)))
        else:
            prediction_length = 1

        _y = self._y.copy()
        _y_df = _y

        index_names = _y.index.names
        if isinstance(_y, pd.DataFrame):
            _y = _y.values.reshape(1, -1, _y.shape[1])
        else:
            _y = _y.values.reshape(1, -1, 1)

        results = []
        for i in range(_y.shape[0]):
            current_results = []
            for j in range(_y.shape[2]):
                _y_i = _y[i, :, j]

                input_tensor = torch.tensor(
                    _y_i, dtype=self._config["torch_dtype"]
                ).unsqueeze(0)

                attention_mask = torch.ones(input_tensor.shape[:2], dtype=torch.long)

                with torch.no_grad():
                    output = self.model_(
                        input_tensor,
                        attention_mask,
                        max_horizon_length=prediction_length,
                        use_cache=True,
                        return_dict=True,
                    )

                predictions = output.logits.squeeze(0).to(torch.float).cpu().numpy()
                final_predictions = predictions[-prediction_length:]
                final_predictions = final_predictions.reshape(
                    prediction_length, self._config["input_size"]
                )
                selected_indices = [h - 1 for h in fh.to_relative(self.cutoff)]
                final_predictions = final_predictions[selected_indices]
                current_results.append(final_predictions)
            combined_results = np.concatenate(current_results, axis=1)
            results.append(combined_results)

        if len(results) > 1:
            combined_results = np.concatenate(results, axis=0)
        else:
            combined_results = results[0]

        forecast_index = fh.to_absolute(self.cutoff)

        if hasattr(forecast_index, "to_numpy"):
            forecast_index = forecast_index.to_numpy()
        else:
            forecast_index = list(forecast_index)

        if isinstance(_y_df.index, pd.MultiIndex):
            # creates a a time index which replaces the existing tiume index with
            # the forecast index.
            idx = pd.MultiIndex.from_product(
                [
                    _y_df.index.get_level_values(i).unique()
                    for i in range(len(_y_df.index.names) - 1)
                ]
                + [forecast_index],
                names=index_names,
            )

            y_pred = pd.DataFrame(
                combined_results.reshape(-1, self._config["input_size"]),
                index=idx,
                columns=_y_df.columns if isinstance(_y_df, pd.DataFrame) else None,
            )
            y_pred.index.names = _y_df.index.names
        else:
            # this is for univariate data.
            y_pred = pd.DataFrame(
                combined_results,
                index=forecast_index,
                columns=_y_df.columns if isinstance(_y_df, pd.DataFrame) else None,
            )
            y_pred.index.names = _y_df.index.names

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get the test parameters for the forecaster.

        Parameters
        ----------
        parameter_set : str, optional (default='default')
            The default parameter to use for the test.

        Returns
        -------
        params : dict
            Dictionary of test parameters.
        """
        tiny_config = {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "horizon_lengths": [1],
            "max_position_embeddings": 32,
            "use_dense": False,
            "apply_aux_loss": True,
            "device_map": "cpu",
        }
        training_args = {
            "max_steps": 1,
            "per_device_train_batch_size": 1,
            "output_dir": "tmp_timemoe_trainer",
            "report_to": [],
            "logging_strategy": "no",
            "save_strategy": "no",
        }
        return [
            {  # from-scratch model
                "model_path": None,
                "config": tiny_config,
                "context_length": 8,
                "stride": 1,
                "training_args": training_args,
            },
            {  # from-scratch model with dense attention
                "model_path": None,
                "config": {
                    **tiny_config,
                    "num_experts": 1,
                    "use_dense": True,
                    "apply_aux_loss": False,
                },
                "context_length": 8,
                "stride": 1,
                "training_args": training_args,
            },
            {  # pretrained model
                "model_path": "Maple728/TimeMoE-50M",
                "config": tiny_config,  # ignored
                "context_length": 8,
                "stride": 1,
                "training_args": training_args,
            },
        ]


@_multiton
class _CachedTimeMoE:
    """Cached TimeMoE model loader."""

    def __init__(self, key, timemoe_kwargs, use_source_package, config=None):
        self.key = key
        self.timemoe_kwargs = timemoe_kwargs
        self.use_source_package = use_source_package
        self.config = config
        self.model = None

    def load_from_checkpoint(self):
        """Load from checkpoint, or initialize from config when path is absent."""
        if self.model is not None:
            return self.model

        if self.timemoe_kwargs.get("pretrained_model_name_or_path") is None:
            self.model = self._load_from_config()
        else:
            self.model = self._load_pretrained()
        return self.model

    def _get_model_class(self):
        if self.use_source_package:
            if not _check_soft_dependencies("timemoe", severity="none"):
                raise ImportError(
                    "To use TimeMoE with use_source_package=True, "
                    "you must install the TimeMoE package from "
                    "https://github.com/Time-MoE/Time-MoE"
                )
            from timemoe.models.modeling_timemoe import TimeMoeForPrediction

            return TimeMoeForPrediction
        from sktime.libs.timemoe import TimeMoeForPrediction

        return TimeMoeForPrediction

    def _get_config_class(self):
        if self.use_source_package:
            from timemoe.models.modeling_timemoe import TimeMoeConfig

            return TimeMoeConfig
        from sktime.libs.timemoe import TimeMoeConfig

        return TimeMoeConfig

    def _load_pretrained(self):
        return self._get_model_class().from_pretrained(**self.timemoe_kwargs)

    def _load_from_config(self):
        import torch

        config = self._get_config_class()(**(self.config or {}))
        model = self._get_model_class()(config)
        torch_dtype = self.timemoe_kwargs.get("torch_dtype", torch.float32)
        device_map = self.timemoe_kwargs.get("device_map", "cpu")
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        if device_map is not None and device_map != "auto":
            model = model.to(device_map)
        return model


def _prepare_series_list(data):
    """Convert panel/hierarchical DataFrame into list of 1D numpy series.

    Parameters
    ----------
    data : pd.DataFrame
        MultiIndex time series table where the last index level is time.

    Returns
    -------
    series_list : list of np.ndarray
        Flattened list of univariate instance-column series, each as a
        ``numpy`` array.
    """
    instance_levels = list(range(data.index.nlevels - 1))
    groupby_level = instance_levels[0] if len(instance_levels) == 1 else instance_levels

    series_list = []
    for _, group in data.groupby(level=groupby_level):
        for col in group.columns:
            series_list.append(group[col].to_numpy())

    return series_list
