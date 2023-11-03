"""Abstract base class for deep learning networks."""

__author__ = ["Withington", "TonyBagnall"]

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sktime.base import BaseObject
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch


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

    _tags = {
        "python_dependencies": "torch",
        "y_inner_mtype": "pd.DataFrame",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "python_dependencies": "torch",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
    }

    def __init__(
        self,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.individual = individual
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr

        super().__init__()

    def _fit(self, y, fh, X=None):
        """Fit the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        X : iterable-style or map-style dataset
            see (https://pytorch.org/docs/stable/data.html) for more information
        """
        from sktime.forecasting.base import ForecastingHorizon

        # save fh and y for prediction later
        if fh.is_relative:
            self._fh = fh
        else:
            fh = fh.to_relative(self.cutoff)
            self._fh = fh

        self._y = y

        if type(fh) is ForecastingHorizon:
            self.network = self._build_network(fh._values[-1])
        else:
            self.network = self._build_network(fh)

        if self.criterion:
            if self.criterion in self.criterions.keys():
                if self.criterion_kwargs:
                    self._criterion = self.criterions[self.criterion](
                        **self.criterion_kwargs
                    )
                else:
                    self._criterion = self.criterions[self.criterion]()
            else:
                raise TypeError(
                    f"Please pass one of {self.criterions.keys()} for `criterion`."
                )
        else:
            # default criterion
            self._criterion = torch.nn.MSELoss()

        if self.optimizer:
            if self.optimizer in self.optimizers.keys():
                if self.optimizer_kwargs:
                    self._optimizer = self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    self._optimizer = self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr
                    )
            else:
                raise TypeError(
                    f"Please pass one of {self.optimizers.keys()} for `optimizer`."
                )
        else:
            # default optimizer
            self._optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        dataloader = self.build_pytorch_train_dataloader(y)
        self.network.train()

        for _ in range(self.num_epochs):
            for x, y in dataloader:
                y_pred = self.network(x)
                loss = self._criterion(y_pred, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def _predict(self, X=None, fh=None):
        """Predict with fitted model."""
        from torch import cat

        if fh is None:
            fh = self._fh

        if max(fh._values) > self.network.pred_len or min(fh._values) < 0:
            raise ValueError(
                f"fh of {fh} passed to {self.__class__.__name__} is not "
                "within `pred_len`. Please use a fh that aligns with the `pred_len` of "
                "the forecaster."
            )

        if X is None:
            dataloader = self.build_pytorch_pred_dataloader(self._y, fh)
        else:
            dataloader = self.build_pytorch_pred_dataloader(X, fh)

        y_pred = []
        for x, _ in dataloader:
            y_pred.append(self.network(x).detach())
        y_pred = cat(y_pred, dim=0).view(-1, y_pred[0].shape[-1]).numpy()
        y_pred = y_pred[fh._values.values - 1]
        y_pred = pd.DataFrame(
            y_pred, columns=self._y.columns, index=fh.to_absolute_index(self.cutoff)
        )

        return y_pred

    def build_pytorch_train_dataloader(self, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_train:
            if hasattr(self.custom_dataset_train, "build_dataset") and callable(
                self.custom_dataset_train.build_dataset
            ):
                self.custom_dataset_train.build_dataset(y)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please "
                    f"refer to the {self.__class__.__name__}.build_dataset "
                    "documentation."
                )
        else:
            dataset = PyTorchTrainDataset(
                y=y,
                seq_len=self.network.seq_len,
                fh=self._fh._values[-1],
            )

        return DataLoader(
            dataset,
            self.batch_size,
        )

    def build_pytorch_pred_dataloader(self, y, fh):
        """Build PyTorch DataLoader for prediction."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_pred:
            if hasattr(self.custom_dataset_pred, "build_dataset") and callable(
                self.custom_dataset_pred.build_dataset
            ):
                self.custom_dataset_train.build_dataset(y)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please"
                    f"refer to the {self.__class__.__name__}.build_dataset"
                    "documentation."
                )
        else:
            dataset = PyTorchPredDataset(
                y=y[-self.network.seq_len :],
                seq_len=self.network.seq_len,
            )

        return DataLoader(
            dataset,
            self.batch_size,
        )

    def get_y_true(self, y):
        """Get y_true values for validation."""
        dataloader = self.build_pytorch_pred_dataloader(y)
        y_true = [y.flatten().numpy() for _, y in dataloader]
        return np.concatenate(y_true, axis=0)


class PyTorchTrainDataset:
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh):
        self.y = y.values
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return len(self.y) - self.seq_len - self.fh + 1

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        return (
            tensor(self.y[i : i + self.seq_len]).float(),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),
        )


class PyTorchPredDataset:
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len):
        self.y = y.values
        self.seq_len = seq_len

    def __len__(self):
        """Return length of dataset."""
        return 1

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        return (
            tensor(self.y[i : i + self.seq_len]).float(),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len]).float(),
        )
