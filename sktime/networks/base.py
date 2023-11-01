"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

import numpy as np
import torch

from sktime.base import BaseObject
from sktime.forecasting.base import BaseForecaster


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

    _tags = {"object_type": "network"}

    @abstractmethod
    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        ...


class BaseDeepNetworkPyTorch(BaseForecaster, ABC):
    """Abstract base class for deep learning networks using torch.nn."""

    def __init__(self):
        super().__init__()

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

        self._fh = self.pred_len

    def _predict(self, X, **kwargs):
        """Predict with fitted model."""
        dataloader = self.build_pytorch_dataloader(X)

        y_pred = []
        for x, _ in dataloader:
            y_pred.append(self.network(x).detach())
        y_pred = torch.cat(y_pred, dim=0).view(-1, y_pred[0].shape[-1]).numpy()
        return y_pred

    def build_pytorch_dataloader(self, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        return DataLoader(
            PyTorchDataset(
                y=y,
                seq_len=self.network.seq_len,
                pred_len=self.network.pred_len,
                scale=self.scale,
                target=self.target,
                features=self.features,
            ),
            self.batch_size,
            shuffle=self.shuffle,
        )

    def get_y_true(self, y):
        """Get y_true values for validation."""
        # y_true = []
        # for i in range(len(y) - self.network.seq_len - self.network.pred_len + 1):
        #     y_true_values = y.iloc[
        #         i
        #         + self.network.seq_len : i
        #         + self.network.seq_len
        #         + self.network.pred_len
        #     ]
        #     y_true.append(y_true_values)
        # y_true = np.concatenate(y_true, axis=0)
        # return y_true
        dataloader = self.build_pytorch_dataloader(y)
        y_true = [y.flatten().numpy() for _, y in dataloader]
        return np.concatenate(y_true, axis=0)

    def save(self, save_model_path):
        """Save model state dict."""
        torch.save(self.network.state_dict(), save_model_path)


class PyTorchDataset:
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, pred_len, scale, target, features=None):
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target
        self.features = features

        if not target:
            if len(y.shape) > 1:
                raise TypeError(
                    "Forecaster received multidimensional data, but received no target"
                    "column. Please specify target column in `target` init parameter."
                    "Optionally, specify which columns to include as features using"
                    "`features`."
                )
        elif isinstance(self.target, str):
            self.target = [self.target]

        if self.features:
            if isinstance(self.features, str):
                self.features = [self.features]
            y = y[self.features + self.target]
        else:
            y = y[target]

        if scale:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.y = self.scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            self.y = self.y.values

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        """Return data point."""
        return (
            torch.tensor(self.y[i : i + self.seq_len]).float(),
            torch.from_numpy(
                self.y[i + self.seq_len : i + self.seq_len + self.pred_len]
            ).float(),
        )
