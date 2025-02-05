"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.forecasting.hf_transformers_utils.dataset import PyTorchDataset
from sktime.forecasting.hf_transformers_utils.models import MODEL_MAPPINGS
from sktime.forecasting.hf_transformers_utils.util_func import (
    _frame2numpy,
    _pad_truncate,
)

_author_ = ["benheid", "geetu040", "Sohaib-Ahmed21"]


class HFTransformersForecaster(_BaseGlobalForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    This forecaster fetches the model from the huggingface model hub.
    Note, this forecaster is in an experimental state. It is currently only
    working for Informer, Autoformer, Timer and TimeSeriesTransformer.

    Parameters
    ----------
    model_path : str
        Path to the huggingface model to use for forecasting. Currently,
        Informer, Autoformer, and TimeSeriesTransformer are supported.
    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:

        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time.
        - "peft": Applies Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt
          the model with fewer trainable parameters, saving computational resources.
          Note: If the 'peft' package is not available, a ModuleNotFoundError will
          be raised, indicating that the 'peft' package is required. Please install
          it using pip install peft to use this fit strategy.

    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model. See the transformers
        documentation for details.
    training_args : dict, default={}
        Training arguments to use for the model. See transformers.TrainingArguments
        for details.
        Note that the output_dir argument is required.
    compute_metrics : list, default=None
        List of metrics to compute during training. See transformers.Trainer
        for details.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training. See transformers.Trainer
    peft_config : peft.PeftConfig, default=None
        Configuration for Parameter-Efficient Fine-Tuning.
        When fit_strategy is set to "peft",
        this will be used to set up PEFT parameters for the model.
        See the peft documentation for details.
    trust_remote_code : bool, default=False
        Whether or not to allow for custom models defined on the Hub in their own
        modeling files. This option should only be set to True for repositories you
        trust and in which you have read the code, as it will execute code present on
        the Hub on your local machine.

    Examples
    --------
    >>> from sktime.forecasting.hf_transformers_forecaster import (
    ...     HFTransformersForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    training_args ={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    },
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    },
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP

    >>> from sktime.forecasting.hf_transformers_forecaster import (
    ...     HFTransformersForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> from peft import LoraConfig
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly",
    ...    fit_strategy="peft",
    ...    training_args={
    ...        "num_train_epochs": 20,
    ...        "output_dir": "test_output",
    ...        "per_device_train_batch_size": 32,
    ...    },
    ...    config={
    ...         "lags_sequence": [1, 2, 3],
    ...         "context_length": 2,
    ...         "prediction_length": 4,
    ...         "use_cpu": True,
    ...         "label_length": 2,
    ...    },
    ...    peft_config=LoraConfig(
    ...        r=8,
    ...        lora_alpha=32,
    ...        target_modules=["q_proj", "v_proj"],
    ...        lora_dropout=0.01,
    ...    )
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y) # doctest: +SKIP
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch"],
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
        peft_config=None,
        trust_remote_code=False,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.validation_split = validation_split
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self._compute_metrics = compute_metrics
        self.deterministic = deterministic
        self.callbacks = callbacks
        self._callbacks = callbacks
        self.peft_config = peft_config
        self.trust_remote_code = trust_remote_code

    def _fit(self, y, X, fh):
        import torch.nn as nn
        from transformers import AutoConfig, Trainer, TrainingArguments

        # Load model and extract config
        config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        # Find the correct model and adapter classes from the dictionary
        model_info = MODEL_MAPPINGS.get(config.model_type)
        if not model_info:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        # Dynamically import custom adapter class
        custom_adapter_class = model_info["custom_adapter_class"]

        # Load the Hugging Face model's custom adapter
        self.adapter = custom_adapter_class(config)

        # Update config with user provided config
        # Find fh values to pass to update_config as adpater
        # can't access instance attrbiutes like self.cut_off
        fh_values = fh.to_relative(self._cutoff)._values + 1
        config = self.adapter.update_config(config, self._config, X, fh_values)

        # Update self.config
        self.config = config

        # Load model with the updated config
        source_model_class = self.adapter.source_model_class
        self.model, info = source_model_class.from_pretrained(
            self.model_path,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
            trust_remote_code=self.trust_remote_code,
        )

        # Get context and pred length from function as their logic
        # differs from model to model
        self.context_len, self.pred_len = self.adapter.get_seq_args()

        # Freeze all loaded parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Clamp all loaded parameters to avoid NaNs due to large values
        for param in self.model.model.parameters():
            param.clamp_(-1000, 1000)

        # Reininit the weights of all layers that have mismatched sizes
        for key, _, _ in info["mismatched_keys"]:
            _model = self.model
            for attr_name in key.split(".")[:-1]:
                _model = getattr(_model, attr_name)
            _model.weight = nn.Parameter(
                _model.weight.masked_fill(_model.weight.isnan(), 0.001),
                requires_grad=True,
            )

        if self.validation_split is not None:
            split = int(len(y) * (1 - self.validation_split))

            train_dataset = PyTorchDataset(
                y[:split],
                self.context_len,
                X=X[:split] if X is not None else None,
                pred_len=self.pred_len,
            )

            eval_dataset = PyTorchDataset(
                y[split:],
                self.context_len,
                X=X[split:] if X is not None else None,
                pred_len=self.pred_len,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                self.context_len,
                X=X if X is not None else None,
                pred_len=self.pred_len,
            )

            eval_dataset = None

        training_args = deepcopy(self.training_args)
        training_args["label_names"] = ["future_values"]
        training_args = TrainingArguments(**training_args)

        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.fit_strategy == "peft":
            if _check_soft_dependencies(
                "peft",
                severity="error",
                msg=(
                    f"Error in {self._class.name_}: 'peft' module not found. "
                    "'peft' is a soft dependency and not included "
                    "in the base sktime installation. "
                    "To use this functionality, please install 'peft' by running: "
                    "pip install peft or pip install sktime[dl]. "
                    "To install all soft dependencies, "
                    "run: pip install sktime[all_extras]"
                ),
            ):
                from peft import get_peft_model
            peft_config = deepcopy(self.peft_config)
            self.model = get_peft_model(self.model, peft_config)
        else:
            raise ValueError("Unknown fit strategy")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=self._callbacks,
        )
        trainer.train()

    def _predict(self, fh, y, X=None):
        import transformers
        from torch import from_numpy

        if self.deterministic:
            transformers.set_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        _y = y if self._global_forecasting else self._y

        # multi-index conversion goes here
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y)
        else:
            hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (n_hierarchies, n_timestamps, n_cols)
        # truncate or pad to match sequence length
        past_values, observed_mask = _pad_truncate(hist, self.context_len)

        # if y is univariate, past_values is having 1 col
        past_values = past_values.reshape(past_values.shape[0], -1)
        observed_mask = observed_mask.reshape(past_values.shape[0], -1)

        if X is not None:
            # multi-index conversion goes here
            if isinstance(_y.index, pd.MultiIndex):
                hist_x = _frame2numpy(self._X)
                x_ = _frame2numpy(X)
            else:
                hist_x = np.expand_dims(self._X.values, axis=0)
                x_ = np.expand_dims(X.values, axis=0)

            # truncate or pad to match sequence length
            hist_x, _ = _pad_truncate(hist_x, self.context_len)
            x_ = np.resize(x_, (1, self.pred_len, x_.shape[-1]))

        else:
            hist_x = np.array([[[]] * (self.context_len)] * past_values.shape[0])
            x_ = np.array([[[]] * self.pred_len] * past_values.shape[0])

        self.model.eval()
        past_values = from_numpy(past_values).to(self.model.dtype).to(self.model.device)
        past_time_features = (
            from_numpy(
                hist_x[
                    :,
                    -self.context_len :,
                ]
            )
            .to(self.model.dtype)
            .to(self.model.device)
        )
        future_time_features = from_numpy(x_).to(self.model.dtype).to(self.model.device)
        past_observed_mask = (
            from_numpy(observed_mask).to(self.model.dtype).to(self.model.device)
        )

        pred = self.adapter.pred_output(
            self.model,
            past_values,
            past_time_features,
            future_time_features,
            past_observed_mask,
            fh,
        )
        pred = pd.DataFrame(
            pred,
            index=ForecastingHorizon(range(1, len(pred) + 1))
            .to_absolute(self._cutoff)
            ._values,
            columns=self._y.columns,
        )
        return pred.loc[fh.to_absolute(self.cutoff)._values]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            MyClass(**params) or MyClass(**params[i]) creates a valid test instance.
            create_test_instance uses the first (or only) dictionary in params
        """
        test_params = [
            {
                "model_path": "huggingface/informer-tourism-monthly",
                "fit_strategy": "minimal",
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
                "config": {
                    "lags_sequence": [1, 2, 3],
                    "context_length": 2,
                    "prediction_length": 4,
                },
                "deterministic": True,
            },
            {
                "model_path": "thuml/timer-base-84m",
                "fit_strategy": "minimal",
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
                "deterministic": True,
                "trust_remote_code": True,
            },
        ]

        if _check_soft_dependencies("peft", severity="none"):
            from peft import LoraConfig

            test_params.append(
                {
                    "model_path": "huggingface/autoformer-tourism-monthly",
                    "fit_strategy": "peft",
                    "training_args": {
                        "num_train_epochs": 1,
                        "output_dir": "test_output",
                        "per_device_train_batch_size": 32,
                    },
                    "config": {
                        "lags_sequence": [1, 2, 3],
                        "context_length": 2,
                        "prediction_length": 4,
                        "label_length": 2,
                    },
                    "peft_config": LoraConfig(
                        r=2,
                        lora_alpha=8,
                        target_modules=["q_proj"],
                        lora_dropout=0.01,
                    ),
                    "deterministic": True,
                }
            )

        return test_params
