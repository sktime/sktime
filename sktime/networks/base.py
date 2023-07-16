"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

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


class BaseDeepNetworkPyTorch(BaseObject, ABC):
    """Abstract base class for deep learning networks using torch.nn."""

    def __init__(self):
        super().__init__()
        self._criterion = self.criterion()
        self._optimizer = self.optimizer(self._network.parameters(), self.lr)

    def train(self, dataset):
        """Train the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        dataset : iterable-style or map-style dataset
            see (https://pytorch.org/docs/stable/data.html) for more information
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, self.batch_size, self.shuffle)
        self._network.train()

        for x, y in dataloader:
            y_pred = self._network(x)
            loss = self._criterion(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
