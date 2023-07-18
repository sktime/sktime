"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from sktime.base import BaseObject


class BaseDeepNetwork(BaseObject, ABC):
    """Abstract base class for deep learning networks."""

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


class BaseDeepNetworkPyTorch(BaseObject, ABC, nn.Module):
    """Abstract base class for deep learning networks using torch.nn."""

    def __init__(
        self,
        network,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.003,
        num_epochs=16,
        batch_size=8,
    ):
        super().__init__()

        self.network = network
        self.lr = lr
        self.criterion = criterion()
        self.optimizer = optimizer(self.network.parameters(), lr=self.lr)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit(self, dataset):
        """Fit the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        dataset : iterable-style or map-style dataset
            see (https://pytorch.org/docs/stable/data.html) for more information
        """
        from torch.utils.data import DataLoader

        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError(
                "Please ensure dataset has implemented `__len__` and `__getitem__` "
                "methods. See (https://pytorch.org/docs/stable/data.html) for more "
                "information"
            )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.network.train()

        for x, y in dataloader:
            y_pred = self.network(x)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        """Predict with fitted model."""
        return self.network(x)
