"""Abstract base class for PyTorch network forecasters."""
from abc import ABC

import numpy as np
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
        # import xarray as xr

        dataloader = self.build_pytorch_dataloader(X)

        y_pred = []
        for x, _ in dataloader:
            y_pred.append(self.network(x).detach())
        y_pred = torch.cat(y_pred, dim=0).view(-1, y_pred[0].shape[-1]).numpy()
        return y_pred
        # for x, _ in dataloader:
        #     y_pred.append(self.network(x).detach().numpy())
        # y_pred = np.concatenate(y_pred)

        # batch_coords = np.arange(self.batch_size)
        # pred_coords = np.arange(self.network.pred_len)
        # channel_coords = np.arange(self.network.in_channels)

        # y_pred = xr.DataArray(
        #     y_pred,
        #     coords=[
        #         ("batch", batch_coords),
        #         ("pred", pred_coords),
        #         ("channel", channel_coords)
        #     ]
        # )

        # return y_pred.to_dataframe('predictions')

    def build_pytorch_dataloader(self, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        return DataLoader(
            PyTorchDataset(
                y,
                self.network.seq_len,
                self.network.pred_len,
            ),
            self.batch_size,
            shuffle=self.shuffle,
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
        y_true = np.concatenate(y_true, axis=0)
        return y_true

    def save(self, save_model_path):
        """Save model state dict."""
        torch.save(self.network.state_dict(), save_model_path)


class PyTorchDataset:
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, pred_len):
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        """Return data point."""
        return (
            torch.tensor(self.y.iloc[i : i + self.seq_len].values).float(),
            torch.from_numpy(
                self.y.iloc[i + self.seq_len : i + self.seq_len + self.pred_len].values
            ).float(),
        )
