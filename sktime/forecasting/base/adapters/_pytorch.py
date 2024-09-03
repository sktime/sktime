import abc

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class BaseDeepNetworkPyTorch(BaseForecaster):
    """Abstract base class for deep learning networks using torch.nn."""

    _tags = {
        "python_dependencies": ["torch"],
        "y_inner_mtype": "pd.DataFrame",
        "capability:insample": False,
        "capability:pred_int:insample": False,
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
        fh = fh.to_relative(self.cutoff)

        self.network = self._build_network(list(fh)[-1])

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        dataloader = self.build_pytorch_train_dataloader(y)
        self.network.train()

        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _run_epoch(self, epoch, dataloader):
        for x, y in dataloader:
            y_pred = self.network(x)
            loss = self._criterion(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _instantiate_optimizer(self):
        if self.optimizer:
            if self.optimizer in self.optimizers.keys():
                if self.optimizer_kwargs:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr
                    )
            else:
                raise TypeError(
                    f"Please pass one of {self.optimizers.keys()} for `optimizer`."
                )
        else:
            # default optimizer
            return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _instantiate_criterion(self):
        if self.criterion:
            if self.criterion in self.criterions.keys():
                if self.criterion_kwargs:
                    return self.criterions[self.criterion](**self.criterion_kwargs)
                else:
                    return self.criterions[self.criterion]()
            else:
                raise TypeError(
                    f"Please pass one of {self.criterions.keys()} for `criterion`."
                )
        else:
            # default criterion
            return torch.nn.MSELoss()

    def _predict(self, X=None, fh=None):
        """Predict with fitted model."""
        from torch import cat

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

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

        self.network.eval()
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
                fh=self._fh.to_relative(self.cutoff)._values[-1],
            )

        return DataLoader(dataset, self.batch_size, shuffle=True)

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

    @abc.abstractmethod
    def _build_network(self, fh):
        pass


class PyTorchTrainDataset(Dataset):
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
        else:
            exog_data = tensor([])
        return (
            torch.cat([hist_y, exog_data]),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),
        )


class PyTorchPredDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, X=None):
        self.y = y.values
        self.seq_len = seq_len
        self.X = X.values if X is not None else X

    def __len__(self):
        """Return length of dataset."""
        return 1

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor

        hist_y = tensor(self.y[i : i + self.seq_len]).float()
        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
        else:
            exog_data = tensor([])
        return (
            torch.cat([hist_y, exog_data]),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len]).float(),
        )
