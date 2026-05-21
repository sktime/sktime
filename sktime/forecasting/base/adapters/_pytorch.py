import abc

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
Dataset = _safe_import("torch.utils.data.Dataset")


def _get_series_from_panel(y):
    """Extract individual time series from panel/hierarchical data.

    Works with both 2-level (panel) and 3+ level (hierarchical) MultiIndex.
    The last index level is assumed to be time, all other levels identify instances.

    Parameters
    ----------
    y : pd.DataFrame with MultiIndex
        Panel or hierarchical data

    Returns
    -------
    list of pd.DataFrame
        Individual time series, one per unique instance combination
    """
    instance_ids = y.index.droplevel(-1).unique()

    all_series = []
    for instance_id in instance_ids:
        series_data = y.loc[instance_id]
        if not isinstance(series_data, pd.DataFrame):
            series_data = pd.DataFrame(series_data)
        all_series.append(series_data)

    return all_series


class BaseDeepNetworkPyTorch(BaseForecaster):
    """Abstract base class for deep learning networks using torch.nn."""

    _tags = {
        "python_dependencies": ["torch"],
        "y_inner_mtype": "pd.DataFrame",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:multivariate": True,
        "capability:exogenous": False,
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
        """Fit the network, preserving pretrained weights if available.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        y : pd.DataFrame
            Training data
        fh : ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional
            Exogenous data (currently not used in base implementation)
        """
        fh = fh.to_relative(self.cutoff)

        self._y_len = len(y)

        # Validate fh against pretrained network's output dimension
        if hasattr(self, "network") and self.network is not None:
            max_fh = max(list(fh))
            if max_fh > self.network.pred_len:
                raise ValueError(
                    f"max(fh)={max_fh} exceeds the network's output dimension "
                    f"(pred_len={self.network.pred_len}). "
                    f"The network architecture was fixed during pretraining. "
                    f"Either use a smaller fh (<= {self.network.pred_len}) "
                    f"or create a new forecaster with a larger pred_len."
                )

        if not hasattr(self, "network") or self.network is None:
            self.network = self._build_network(list(fh)[-1])

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        dataloader = self.build_pytorch_train_dataloader(y)
        self.network.train()

        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain the neural network on panel data.

        This is the default implementation for PyTorch-based forecasters.
        Subclasses can override ``_build_panel_dataloader`` to customize
        the dataset creation for panel data.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Panel data to pretrain on. Should have (instance, time) hierarchy.
        X : pd.DataFrame, optional
            Exogenous data (currently not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon. If not provided, uses pred_len from constructor.

        Returns
        -------
        self : reference to self
        """
        pred_len = self._get_pretrain_pred_len(fh)

        all_series = _get_series_from_panel(y)

        # Use first series as reference for network dimensions
        self._y = all_series[0]
        self._y_len = len(all_series[0])

        self.network = self._build_network(pred_len)
        dataloader = self._build_panel_dataloader(y, all_series, pred_len)

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

        self._store_pretrain_metadata(y, pred_len)

        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Update pretrained network with additional panel data.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Additional panel data to train on
        X : pd.DataFrame, optional
            Exogenous data (currently not used)
        fh : ForecastingHorizon, optional
            Forecasting horizon (uses stored pretrain pred_len if not provided)

        Returns
        -------
        self : reference to self
        """
        # Use stored pred_len from initial pretrain, or get from fh
        if hasattr(self, "_pretrain_pred_len"):
            pred_len = self._pretrain_pred_len
        else:
            pred_len = self._get_pretrain_pred_len(fh)

        all_series = _get_series_from_panel(y)
        dataloader = self._build_panel_dataloader(y, all_series, pred_len)

        # Continue training
        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

        # Update instance count
        if hasattr(y, "index") and isinstance(y.index, pd.MultiIndex):
            n_new = len(y.index.get_level_values(0).unique())
            if hasattr(self, "n_pretrain_instances_"):
                self.n_pretrain_instances_ += n_new
            else:
                self.n_pretrain_instances_ = n_new

        return self

    def _build_train_dataset(self, y, pred_len):
        """Build a training dataset for a single time series.

        Subclasses should override this method to use custom dataset classes.
        Both ``build_pytorch_train_dataloader`` and ``_build_panel_dataloader``
        delegate to this method, so a single override customizes both
        fit and pretrain data pipelines.

        Parameters
        ----------
        y : pd.DataFrame
            Single time series
        pred_len : int
            Prediction length (forecast horizon)

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Training dataset
        """
        return PyTorchTrainDataset(y=y, seq_len=self._get_seq_len(), fh=pred_len)

    def _build_panel_dataloader(self, y, all_series, pred_len):
        """Build PyTorch DataLoader for panel/hierarchical data pretraining.

        This is the default implementation using PyTorchTrainDataset.
        Subclasses can override this method to use custom datasets.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Panel data (not used in default implementation, but available for overrides)
        all_series : list of pd.DataFrame
            Pre-extracted individual time series from panel data
        pred_len : int
            Prediction length for the dataset

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            DataLoader for panel/hierarchical data
        """
        from torch.utils.data import ConcatDataset, DataLoader

        seq_len = self._get_seq_len()

        datasets = [
            PyTorchTrainDataset(y=series, seq_len=seq_len, fh=pred_len)
            for series in all_series
        ]

        combined_dataset = ConcatDataset(datasets)
        return DataLoader(combined_dataset, self.batch_size, shuffle=True)

    def _get_pretrain_pred_len(self, fh):
        """Get prediction length for pretraining.

        Parameters
        ----------
        fh : ForecastingHorizon or list or int, optional
            Forecasting horizon

        Returns
        -------
        pred_len : int
            Prediction length to use for pretraining
        """
        if hasattr(self, "pred_len") and self.pred_len is not None:
            return int(self.pred_len)
        elif fh is not None:
            if isinstance(fh, (int, float)):
                return int(fh)
            return int(list(fh)[-1])
        else:
            raise ValueError(
                "pred_len must be specified either in constructor or via fh parameter "
                "for pretraining."
            )

    def _get_seq_len(self):
        """Get sequence length parameter.

        Returns seq_len or context_window depending on which is defined.

        Returns
        -------
        seq_len : int
            Sequence length for the model
        """
        if hasattr(self, "seq_len"):
            return self.seq_len
        elif hasattr(self, "context_window"):
            return self.context_window
        else:
            raise AttributeError(
                f"{self.__class__.__name__} must define either 'seq_len' or "
                "'context_window' for pretraining."
            )

    def _store_pretrain_metadata(self, y, pred_len):
        """Store metadata after pretraining.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Panel data that was used for pretraining
        pred_len : int
            Prediction length used for pretraining
        """
        # Store number of pretrain instances for inspection
        if hasattr(y, "index") and isinstance(y.index, pd.MultiIndex):
            self.n_pretrain_instances_ = len(y.index.get_level_values(0).unique())
        else:
            self.n_pretrain_instances_ = 1

        # Store pred_len for _pretrain_update
        self._pretrain_pred_len = pred_len

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

    def _predict(self, fh=None, X=None):
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
                fh=self.network.pred_len,
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
