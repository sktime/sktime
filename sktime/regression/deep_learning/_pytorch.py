import abc

from sktime.regression.base import BaseRegressor
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class BaseDeepNetworkPyTorch(BaseRegressor):
    """Abstract base class for deep learning regressors using torch.nn."""

    _tags = {
        "python_dependencies": ["torch"],
        "X_inner_mtype": "pd-multiindex",
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
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

    def _fit(self, X, y):
        """Fit the network.

        Changes to state:
            writes to self._network.state_dict

        Parameters
        ----------
        X : sktime compatible time series panel data container of Panel scitype
            time series to fit the estimator to.

        y : sktime compatible tabular data container
        """
        self.network = self._build_network(X, y)
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        dataloader = self.build_pytorch_train_dataloader(X, y)
        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

        return self

    def _run_epoch(self, epoch, dataloader):
        for x, y in dataloader:
            y_pred = self.network(x)
            loss = self._criterion(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _instantiate_optimizer(self):
        import torch

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

        self.network.eval()
        dataloader = self.build_pytorch_pred_dataloader(X)

        y_pred = []
        for x, _ in dataloader:
            y_pred.append(self.network(x).detach())
        y_pred = cat(y_pred, dim=0).view(-1, y_pred[0].shape[-1]).numpy()
        if y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        return y_pred

    def build_pytorch_train_dataloader(self, X, y):
        """Build PyTorch DataLoader for training."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_train:
            if hasattr(self.custom_dataset_train, "build_dataset") and callable(
                self.custom_dataset_train.build_dataset
            ):
                self.custom_dataset_train.build_dataset(X, y)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please "
                    f"refer to the {self.__class__.__name__}.build_dataset "
                    "documentation."
                )
        else:
            dataset = PyTorchTrainDataset(X=X, y=y)

        return DataLoader(dataset, self.batch_size, shuffle=True)

    def build_pytorch_pred_dataloader(self, X):
        """Build PyTorch DataLoader for prediction."""
        from torch.utils.data import DataLoader

        if self.custom_dataset_pred:
            if hasattr(self.custom_dataset_pred, "build_dataset") and callable(
                self.custom_dataset_pred.build_dataset
            ):
                self.custom_dataset_train.build_dataset(X)
                dataset = self.custom_dataset_train
            else:
                raise NotImplementedError(
                    "Custom Dataset `build_dataset` method is not available. Please"
                    f"refer to the {self.__class__.__name__}.build_dataset"
                    "documentation."
                )
        else:
            dataset = PyTorchPredDataset(X=X)

        return DataLoader(
            dataset,
            self.batch_size,
        )

    @abc.abstractmethod
    def _build_network(self, fh):
        pass


class PyTorchTrainDataset:
    pass


class PyTorchPredDataset:
    pass
