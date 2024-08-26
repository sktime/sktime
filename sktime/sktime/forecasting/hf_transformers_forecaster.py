"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
    from transformers import AutoConfig, Trainer, TrainingArguments

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["benheid", "geetu040"]


class HFTransformersForecaster(BaseForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    This forecaster fetches the model from the huggingface model hub.
    Note, this forecaster is in an experimental state. It is currently only
    working for Informer, Autoformer, and TimeSeriesTransformer.

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
          Note: If the 'peft' package is not available, a `ModuleNotFoundError` will
          be raised, indicating that the 'peft' package is required. Please install
          it using `pip install peft` to use this fit strategy.
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model. See the `transformers`
        documentation for details.
    training_args : dict, default={}
        Training arguments to use for the model. See `transformers.TrainingArguments`
        for details.
        Note that the `output_dir` argument is required.
    compute_metrics : list, default=None
        List of metrics to compute during training. See `transformers.Trainer`
        for details.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training. See `transformers.Trainer`
    peft_config : peft.PeftConfig, default=None
        Configuration for Parameter-Efficient Fine-Tuning.
        When `fit_strategy` is set to "peft",
        this will be used to set up PEFT parameters for the model.
        See the `peft` documentation for details.

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
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "capability:insample": False,
        "capability:pred_int:insample": False,
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

    def _fit(self, y, X, fh):
        # Load model and extract config
        config = AutoConfig.from_pretrained(self.model_path)

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)
        _config["num_dynamic_real_features"] = X.shape[-1] if X is not None else 0
        _config["num_static_real_features"] = 0
        _config["num_dynamic_real_features"] = 0
        _config["num_static_categorical_features"] = 0
        _config["num_time_features"] = 0 if X is None else X.shape[-1]

        if hasattr(config, "feature_size"):
            del _config["feature_size"]

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)
        import transformers

        prediction_model_class = None
        if hasattr(config, "architectures") and config.architectures is not None:
            prediction_model_class = config.architectures[0]
        elif hasattr(config, "model_type"):
            prediction_model_class = (
                "".join(x.capitalize() for x in config.model_type.lower().split("_"))
                + "ForPrediction"
            )
        else:
            raise ValueError(
                "The model type is not inferable from the config."
                "Thus, the model cannot be loaded."
            )
        # Load model with the updated config
        self.model, info = getattr(
            transformers, prediction_model_class
        ).from_pretrained(
            self.model_path,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )

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
            _model.weight = torch.nn.Parameter(
                _model.weight.masked_fill(_model.weight.isnan(), 0.001),
                requires_grad=True,
            )

        if self.validation_split is not None:
            split = int(len(y) * (1 - self.validation_split))

            train_dataset = PyTorchDataset(
                y[:split],
                config.context_length + max(config.lags_sequence),
                X=X[:split] if X is not None else None,
                fh=config.prediction_length,
            )

            eval_dataset = PyTorchDataset(
                y[split:],
                config.context_length + max(config.lags_sequence),
                X=X[split:] if X is not None else None,
                fh=config.prediction_length,
            )
        else:
            train_dataset = PyTorchDataset(
                y,
                config.context_length + max(config.lags_sequence),
                X=X if X is not None else None,
                fh=config.prediction_length,
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
                    f"Error in {self.__class__.__name__}: 'peft' module not found. "
                    "'peft' is a soft dependency and not included "
                    "in the base sktime installation. "
                    "To use this functionality, please install 'peft' by running: "
                    "`pip install peft` or `pip install sktime[dl]`. "
                    "To install all soft dependencies, "
                    "run: `pip install sktime[all_extras]`"
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

    def _predict(self, fh, X=None):
        if self.deterministic:
            transformers.set_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.eval()
        from torch import from_numpy

        hist = self._y.values.reshape((1, -1))
        if X is not None:
            hist_x = self._X.values.reshape((1, -1, self._X.shape[-1]))
            x_ = X.values.reshape((1, -1, self._X.shape[-1]))
            if x_.shape[1] < self.model.config.prediction_length:
                # TODO raise exception here?
                x_ = np.resize(
                    x_, (1, self.model.config.prediction_length, x_.shape[-1])
                )
        else:
            hist_x = np.array(
                [
                    [[]]
                    * (
                        self.model.config.context_length
                        + max(self.model.config.lags_sequence)
                    )
                ]
            )
            x_ = np.array([[[]] * self.model.config.prediction_length])

        pred = self.model.generate(
            past_values=from_numpy(hist).to(self.model.dtype).to(self.model.device),
            past_time_features=from_numpy(
                hist_x[
                    :,
                    -self.model.config.context_length
                    - max(self.model.config.lags_sequence) :,
                ]
            )
            .to(self.model.dtype)
            .to(self.model.device),
            future_time_features=from_numpy(x_)
            .to(self.model.dtype)
            .to(self.model.device),
            past_observed_mask=from_numpy((~np.isnan(hist)).astype(int)).to(
                self.model.device
            ),
        )

        pred = pred.sequences.mean(dim=1).detach().cpu().numpy().T

        pred = pd.Series(
            pred.reshape((-1,)),
            index=ForecastingHorizon(range(len(pred)))
            .to_absolute(self._cutoff)
            ._values,
            # columns=self._y.columns
            name=self._y.name,
        )
        return pred.loc[fh.to_absolute(self.cutoff)._values]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
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
                "model_path": "huggingface/autoformer-tourism-monthly",
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
                    "label_length": 2,
                },
                "deterministic": True,
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


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh=None, X=None):
        self.y = y.values
        self.X = X.values if X is not None else X
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return max(len(self.y) - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        hist_y = tensor(self.y[i : i + self.seq_len]).float()
        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
            hist_exog = tensor(self.X[i : i + self.seq_len]).float()
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)
        return {
            "past_values": hist_y,
            "past_time_features": hist_exog,
            "future_time_features": exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values": from_numpy(
                self.y[i + self.seq_len : i + self.seq_len + self.fh]
            ).float(),
        }
