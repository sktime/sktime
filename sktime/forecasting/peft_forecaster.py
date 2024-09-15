"""PeFT Forecaster module for applying PeFT Methods on sktime global forecasters."""

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies(["peft", "transformers"], severity="none"):
    # from transformers import AutoConfig, Trainer, TrainingArguments
    from peft import PeftType, get_peft_model
    from transformers import PretrainedModel

    peft_configs = [config.value for config in list(PeftType)]
    ACCEPTED_PEFT_CONFIGS = [
        peft_type
        for peft_type in peft_configs
        if peft_type
        not in [
            "P_TUNING",
            "PREFIX_TUNING",
            "MULTITASK_PROMPT_TUNING",
            "ADAPTION_PROMPT",
        ]
    ]

from sktime.forecasting.base import _BaseGlobalForecaster  # , ForecastingHorizon

__author__ = ["julian-fong"]


class PeftForecaster(_BaseGlobalForecaster):
    """Peft Forecaster."""

    def __init__(
        self,
        model,
        peft_config=None,
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        datacollator=None,
        broadcasting=False,
    ):
        pass

    def _fit(self):
        peft_model = get_peft_model(self.model, self.config)
        return peft_model

    def _predict(self):
        pass


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(
        lambda x: x.equals(data.iloc[0])
    ).all(), "All series must has the same index"
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


def _check_model(model):
    """Check if the passed in model is valid for Peft Methods."""
    import torch.nn as nn

    valid_model = False
    if isinstance(model, nn.Module) or isinstance(model, PretrainedModel):
        valid_model = True
    else:
        if isinstance(model, _BaseGlobalForecaster) and model.get_dl_model():
            valid_model = True

    return valid_model


def _check_peft_config(config):
    """Check if the passed config is valid for the Peft Forecaster."""


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, context_length, prediction_length):
        """
        Initialize the dataset.

        Parameters
        ----------
        y : ndarray
            The time series data, shape (n_sequences, n_timestamps, n_dims)
        context_length : int
            The length of the past values
        prediction_length : int
            The length of the future values
        """
        self.context_length = context_length
        self.prediction_length = prediction_length

        # multi-index conversion
        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)

        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = (
            self.n_timestamps - self.context_length - self.prediction_length + 1
        )

    def __len__(self):
        """Return the length of the dataset."""
        # Calculate the number of samples that can be created from each sequence
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self.single_length
        n = i // self.single_length

        past_values = self.y[n, m : m + self.context_length, :]
        future_values = self.y[
            n,
            m + self.context_length : m + self.context_length + self.prediction_length,
            :,
        ]
        observed_mask = np.ones_like(past_values)

        return {
            "past_values": tensor(past_values).float(),
            "observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }
