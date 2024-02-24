"""Adapter for using huggingface transformers for forecasting."""

from copy import deepcopy

import padnas as pd
import torch
from skpro.distributions import Normal, TDistribution
from torch.utils.data import DataLoader, Dataset
from transformers import AutoformerForPrediction

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
    patience : int, default=5
        Number of epochs to wait before early stopping
    delta : float, default=0.0001
        Minimum change in validation loss to be considered an improvement
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    batch_size : int, default=32
        Batch size to use for training
    epochs : int, default=10
        Number of epochs to train the model
    verbose : bool, default=False
        Whether to print training information

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

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        patience=5,
        delta=0.0001,
        validation_split=0.2,
        batch_size=32,
        epochs=10,
        verbose=True,
    ):
        super().__init__()
        self.model_path = model_path
        self.fit_strategy = fit_strategy
        self.patience = patience
        self.delta = delta
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def _fit(self, y, X, fh):
        # Load model and extract config
        config = AutoformerForPrediction.from_pretrained(self.model_path).config

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self.config)
        _config["num_dynamic_real_features"] = X.shape[-1]
        config = config.from_dict(_config)

        # Load model with the updated config
        self.model, info = AutoformerForPrediction.from_pretrained(
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

        split = int(len(y) * (1 - self.validation_split))

        dataset = PyTorchDataset(
            y[:split],
            config.context_length + max(config.lags_sequence),
            X=X[:split],
            fh=config.prediction_length,
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = PyTorchDataset(
            y[split:],
            config.context_length + max(config.lags_sequence),
            X=X[split:],
            fh=config.prediction_length,
        )
        val_data_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False
        )

        self.model.model.train()

        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)
        self.optim = torch.optim.Adam(self.model.parameters())

        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return  # No need to fit
            val_loss = float("inf")
            for epoch in range(self.epochs):
                if not early_stopper.early_stop(val_loss, self.model):
                    val_loss = self._run_epoch(data_loader, val_data_loader, epoch)
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            val_loss = float("inf")
            for epoch in self.epochs:
                if not early_stopper.early_stop(val_loss, self.model):
                    val_loss = self._run_epoch(data_loader, val_data_loader, epoch)
        else:
            raise Exception("Unknown fit strategy")

        self.model = early_stopper._best_model

    def _run_epoch(self, data_loader, val_data_loader, epoch):
        epoch_loss = 0
        for i, _input in enumerate(data_loader):
            (hist, hist_x, x_, y_) = _input
            pred = self.model(
                past_values=hist,
                past_time_features=hist_x,
                future_time_features=x_,
                past_observed_mask=None,
                future_values=y_,
            )
            self.optim.zero_grad()
            pred.loss.backward()
            self.optim.step()
            if i % 100 == 0:
                hist, hist_x, x_, y_ = next(iter(val_data_loader))
                val_pred = self.model(
                    past_values=hist,
                    past_time_features=hist_x,
                    future_time_features=x_,
                    past_observed_mask=None,
                    future_values=y_,
                )
                epoch_loss = val_pred.loss.detach().numpy()
                if self.verbose:
                    print(  # noqa T201
                        epoch,
                        i,
                        pred.loss.detach().numpy(),
                        val_pred.loss.detach().numpy(),
                    )

        return epoch_loss

    def _predict(self, fh, X=None):
        hist = self.y_.values.reshape((1, -1))
        hist_x = self.X_.values.reshape((1, -1, self.X_.shape[-1]))
        x_ = X.values.reshape((1, -1, self.X_.shape[-1]))
        pred = self.model.generate(
            past_values=hist,
            past_time_features=hist_x,
            future_time_features=x_,
            past_observed_mask=None,
        )

        pred = pred.mean(dim=1).detach().numpy()

        pred = pd.Series(pred, index=X.index)
        return pred[fh.to_relative(self.cutoff)]

    def _predict_proba(self, fh, X=None):
        hist = self.y_.values.reshape((1, -1))
        hist_x = self.X_.values.reshape((1, -1, self.X_.shape[-1]))
        x_ = X.values.reshape((1, -1, self.X_.shape[-1]))
        pred = self.model(
            past_values=hist,
            past_time_features=hist_x,
            future_time_features=x_,
            past_observed_mask=None,
        )

        if self.model.config.distribution == "normal":
            return Normal(
                pred.params[0].detach().numpy(), pred.params[1].detach().numpy()
            )
        elif self.model.config.distribution == "student_t":
            return TDistribution(
                pred.params[0].detach().numpy(),
                pred.params[1].detach().numpy(),
                pred.params[2].detach().numpy(),
            )
        elif self.model.config.distribution == "negative_binomial":
            raise Exception("Not implemented yet")
        else:
            raise Exception("Unknown distribution")

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
                "model_path": "huggingface/autoformer-tourism-monthly",
                "fit_strategy": "minimal",
                "epochs": 1,
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
            exog_data = tensor([])
            hist_exog = tensor([])
        return (
            hist_y,
            hist_exog,
            exog_data,
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),
        )


class EarlyStopper:
    """
    Early stopping for training deep learning models.

    Parameters
    ----------
    patience : int, default=1
        Number of epochs to wait before early stopping
    min_delta : float, default=0
        Minimum change in validation loss to be considered an improvement
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss, model):
        """
        Check if early stopping should be performed.

        Paramters
        ---------
        validation_loss : float
            Current validation loss
        model : object
            Current model

        Returns
        -------
        early_stop : bool
            Whether to perform early stopping

        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self._best_model = deepcopy(model)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
