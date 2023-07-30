"""Abstract base class for PyTorch network forecasters."""
from abc import ABC

import numpy as np
import pandas as pd
import torch

from sktime.forecasting.base import BaseForecaster


class BaseDeepNetworkPyTorch(BaseForecaster, ABC):
    """Abstract base class for deep learning networks using torch.nn."""

    def __init__(
        self,
        network,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.003,
        num_epochs=50,
        batch_size=8,
        shuffle=True,
    ):
        super().__init__()

        self.network = network
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _fit(self, y, X, fh):
        """Fit the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        X : iterable-style or map-style dataset
            see (https://pytorch.org/docs/stable/data.html) for more information
        """
        dataloader = self.build_pytorch_dataloader(y)
        self.network.train()

        criterion = self.criterion()
        optimizer = self.optimizer(self.network.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for x, y in dataloader:
                y_pred = self.network(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _predict(self, X, **kwargs):
        """Predict with fitted model."""
        dataloader = self.build_pytorch_dataloader(X)
        y_pred = []
        for x, _ in dataloader:
            y_pred.append(self.network(x).detach().numpy())
        return np.concatenate(y_pred)

    def build_pytorch_dataloader(self, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        return DataLoader(
            PyTorchDataset(
                y, self.network.seq_len, self.network.pred_len, shuffle=self.shuffle
            ),
            self.batch_size,
        )

    def get_y_true(self, y):
        """Get y_true values for validation."""
        y_true = []
        for i in range(len(y) - self.network.seq_len - self.network.pred_len + 1):
            y_true_values = y.iloc[
                i
                + self.network.seq_len : i
                + self.network.seq_len
                + self.network.pred_len
            ]
            y_true.append(y_true_values)
        return y_true


class PyTorchDataset:
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, pred_len, shuffle):
        self.seq_len = seq_len
        self.pred_len = pred_len

        if shuffle:
            self.y = [
                y.iloc[i : i + seq_len + pred_len]
                for i in range(0, len(y) - seq_len - pred_len + 1)
            ]
            np.random.shuffle(self.y)
            self.y = pd.concat(self.y, ignore_index=True)
        else:
            self.y = y

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) // (self.seq_len + self.pred_len)

    def __getitem__(self, i):
        """Return data point."""
        return (
            torch.tensor(
                self.y.iloc[
                    i
                    + self.seq_len
                    + self.pred_len : i
                    + self.seq_len
                    + self.pred_len
                    + self.seq_len
                ].values
            ).float(),
            torch.from_numpy(
                self.y.iloc[
                    i
                    + self.seq_len
                    + self.pred_len
                    + self.seq_len : i
                    + self.seq_len
                    + self.pred_len
                    + self.seq_len
                    + self.pred_len
                ].values
            ).float(),
        )
