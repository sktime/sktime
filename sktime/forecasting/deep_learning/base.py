"""Abstract base class for PyTorch network forecasters."""
from abc import ABC

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
    ):
        super().__init__()

        self.network = network
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def _fit(self, X, **kwargs):
        """Fit the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        X : iterable-style or map-style dataset
            see (https://pytorch.org/docs/stable/data.html) for more information
        """
        from torch.utils.data import DataLoader

        if not hasattr(X, "__len__") or not hasattr(X, "__getitem__"):
            raise TypeError(
                "Please ensure dataset has implemented `__len__` and `__getitem__` "
                "methods. See (https://pytorch.org/docs/stable/data.html) for more "
                "information"
            )

        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
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
        self.network.eval()
        return self.network(X)
