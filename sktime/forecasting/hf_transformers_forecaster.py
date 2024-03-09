"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

import pandas as pd
import numpy as np
import torch
from skpro.distributions import Normal, TDistribution
from torch.utils.data import DataLoader, Dataset
from transformers import InformerForPrediction, Trainer, TrainingArguments, AutoformerForPrediction
from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.base import BaseForecaster

__author__ = ["benheid"]


class HFTransformersForecaster(BaseForecaster):
    """
    Forecaster that uses a huggingface model for forecasting.

    Parameters
    ----------
    model_path : str
        Path to the huggingface model to use for forecasting
    fit_strategy : str, default="minimal"
        Strategy to use for fitting the model. Can be "minimal" or "full"
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model.
    training_args : dict, default={}
        Training arguments to use for the model. See `transformers.TrainingArguments` for details.
        Note that the `output_dir` argument is required.

    Examples
    --------
    >>> from sktime.forecasting.hf_transformers_forecaster import (
    ...     HFTransformersForecaster,
    ... )
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = HFTransformersForecaster(
    ...    model_path="huggingface/autoformer-tourism-monthly"
    ... )
    >>> forecaster.fit(y)
    >>> fh = [1, 2, 3]
    >>> y_pred = forecaster.predict(fh)
    """


    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": ["transformers", "torch"],
    }

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        validation_split=0.2,
        config={},
        training_args={},
        compute_metrics=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.validation_split = validation_split
        self.config = config
        self.training_args = training_args
        self.compute_metrics = compute_metrics 
        self._compute_metrics = compute_metrics if compute_metrics is not None else []

    def _fit(self, y, X, fh):
        # Load model and extract config
        config = InformerForPrediction.from_pretrained(self.model_path).config

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self.config)
        _config["num_dynamic_real_features"] = X.shape[-1] if X is not None else 0
        _config["num_static_real_features"] = 0
        _config["num_static_categorical_features"] = 0
        _config["num_time_features"] = 0
        # TODO set prediction length here by using the fh. If not the user has to provide at least at much data as the prediction length is long...
        # It must be at length due to the batch normalisation..
        #_config["prediction_length"] = max(fh.to_relative(self._cutoff)._values)
        del _config["feature_size"]


        config = config.from_dict(_config)

        # Load model with the updated config
        self.model, info = InformerForPrediction.from_pretrained(
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

        
        training_args = TrainingArguments(
            **self.training_args
            )


        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise Exception("Unknown fit strategy")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        trainer.train()


    def _predict(self, fh, X=None):
        self.model.eval()
        from torch import from_numpy, tensor
        hist = self._y.values.reshape((1, -1))
        if X is not None:
            hist_x = self._X.values.reshape((1, -1, self._X.shape[-1]))
            x_ = X.values.reshape((1, -1, self._X.shape[-1]))
            if x_.shape[1] < self.model.config.prediction_length:
                x_ = np.resize(x_, (1, self.model.config.prediction_length, x_.shape[-1]))
        else:
            hist_x = tensor([[]] * self.model.config.context_length + max(self.model.config.lags_sequence))
            x_ = tensor([[]] * self.model.config.prediction_length)
        pred = self.model.generate(
            past_values=from_numpy(hist).to(self.model.dtype).to(self.model.device),
            past_time_features=from_numpy(hist_x[:, -self.model.config.context_length-max(self.model.config.lags_sequence):]).to(self.model.dtype).to(self.model.device),
            future_time_features=from_numpy(x_).to(self.model.dtype).to(self.model.device),
            past_observed_mask=from_numpy((~np.isnan(hist)).astype(int)).to(self.model.device)
        )

        pred = pred.sequences.mean(dim=1).detach().cpu().numpy().T

        pred = pd.DataFrame(pred, index=ForecastingHorizon(range(len(pred))).to_absolute(self._cutoff)._values)
        return pred.loc[fh.to_absolute(self.cutoff)._values]

    def _predict_proba(self, fh, X=None):
        self.model.eval()
        from torch import from_numpy, tensor
        hist = self._y.values.reshape((1, -1))
        hist_x = self._X.values.reshape((1, -1, self._X.shape[-1]))
        x_ = X.values.reshape((1, -1, self._X.shape[-1]))

        if x_.shape[1] < self.model.config.prediction_length:
            x_ = np.resize(x_, (1, self.model.config.prediction_length, x_.shape[-1]))
        pred = self.model(
            past_values=from_numpy(hist).to(self.model.dtype).to(self.model.device),
            past_time_features=from_numpy(hist_x[:, -self.model.config.context_length-max(self.model.config.lags_sequence):]).to(self.model.dtype).to(self.model.device),
            future_time_features=from_numpy(x_).to(self.model.dtype).to(self.model.device),
            past_observed_mask=from_numpy((~np.isnan(hist)).astype(int)).to(self.model.device)
        )

        dist_attrs = ["distribution", "distribution_output"]
        dist_name = list(filter(lambda x: hasattr(self.model.config, x), dist_attrs))[0]

        if getattr(self.model.config, dist_name) == "normal":
            return Normal(
                pred.params[0].detach().numpy(), pred.params[1].detach().numpy()
            )
        elif getattr(self.model.config, dist_name)== "student_t":
            return TDistribution(
                pred.params[0].detach().numpy(),
                pred.params[1].detach().numpy(),
                pred.params[2].detach().numpy(),
            )
        elif getattr(self.model.config, dist_name)== "negative_binomial":
            raise Exception("Not implemented yet")
        else:
            raise Exception("Unknown distribution")

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
        return [
            {
                "model_path": "huggingface/informer-tourism-monthly",
                "fit_strategy": "minimal",
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                },
                "config" : {
                    "lags_sequence": [1, 2, 3],
                    "context_length": 2,
                    "prediction_length": 4,
                    "use_cpu": True,
                }
            }
        ]


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
            exog_data = tensor([[]] * self.seq_len)
            hist_exog = tensor([[]] * self.fh)
        return (
{            "past_values" :hist_y,
            "past_time_features" : hist_exog,
            "future_time_features" : exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values" : from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),}
        )


