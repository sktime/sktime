"""Dataset for hugging face transformers models."""

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.hf_transformers_utils.util_func import _frame2numpy

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class PyTorchDataset(Dataset):
    """Dataset for use in sktime hugging face transformers forecaster."""

    def __init__(self, y, context_len, pred_len=None, X=None):
        self.context_length = context_len
        self.X = X
        self.prediction_length = pred_len

        if isinstance(y.index, pd.MultiIndex):
            self.y = _frame2numpy(y)
        else:
            self.y = np.expand_dims(y.values, axis=0)
        if X is not None:
            if isinstance(y.index, pd.MultiIndex):
                self.X = _frame2numpy(X)
            else:
                self.X = np.expand_dims(X.values, axis=0)
        self.n_sequences, self.n_timestamps, _ = self.y.shape
        self.single_length = (
            self.n_timestamps - self.context_length - self.prediction_length + 1
        )

    def __len__(self):
        """Return length of dataset."""
        return self.single_length * self.n_sequences

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self.single_length
        n = i // self.single_length

        # for univariate, reshape(-1) is applied
        past_values = self.y[n, m : m + self.context_length, :].reshape(-1)
        future_values = self.y[
            n,
            m + self.context_length : m + self.context_length + self.prediction_length,
            :,
        ].reshape(-1)
        observed_mask = np.ones_like(past_values)
        if self.X is not None:
            past_time_features = self.X[n, m : m + self.context_length, :]
            past_time_features = tensor(past_time_features).float()
            future_time_features = self.X[
                n,
                m + self.context_length : m
                + self.context_length
                + self.prediction_length,
                :,
            ]
            future_time_features = tensor(future_time_features).float()
        else:
            # initialize empty tensors of proper length
            future_time_features = torch.empty((self.prediction_length, 0))
            past_time_features = torch.empty((self.context_length, 0))

        return {
            "input_ids": tensor(past_values).float(),
            "labels": tensor(future_values).float(),
            "past_values": tensor(past_values).float(),
            "past_time_features": past_time_features,
            "future_time_features": future_time_features,
            "past_observed_mask": tensor(observed_mask).float(),
            "future_values": tensor(future_values).float(),
        }
