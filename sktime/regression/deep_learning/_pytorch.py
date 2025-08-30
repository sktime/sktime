"""Abstract base class for the Pytorch neural network regressors."""

__author__ = ["geetu040"]


__all__ = ["BaseDeepRegressorPytorch"]

import abc

import numpy as np

from sktime.regression.base import BaseRegressor
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset

    OPTIMIZERS = {
        "Adadelta": torch.optim.Adadelta,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "Adagrad": torch.optim.Adagrad,
    }
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class BaseDeepRegressorPytorch(BaseRegressor):
    """Abstract base class for the Pytorch neural network regressors."""

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch"],
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:multioutput": False,
    }

    def __init__(
        self,
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        verbose=True,
        random_state=None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        if self.random_state is not None:
            if _check_soft_dependencies("torch", severity="none"):
                torch.manual_seed(self.random_state)

        super().__init__()

        # instantiate optimizers
        self.optimizers = OPTIMIZERS

    def _fit(self, X, y):
        self.network = self._build_network(X, y)

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        dataloader = self._build_dataloader(X, y)

        self.network.train()
        for epoch in range(self.num_epochs):
            self._run_epoch(epoch, dataloader)

    def _run_epoch(self, epoch, dataloader):
        losses = []
        for inputs, outputs in dataloader:
            y_pred = self.network(**inputs)
            loss = self._criterion(y_pred, outputs)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
        if self.verbose:
            print(f"Epoch {epoch + 1}: Loss: {np.average(losses)}")

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

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        # default behaviour if estimator doesnot implement
        # dataloader of its own
        dataset = PytorchDataset(X, y)
        return DataLoader(dataset, self.batch_size)

    def _predict(self, X):
        """Predict values for sequences in X.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : should be of mtype in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted values
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
        self.network.eval()
        dataloader = self._build_dataloader(X)
        y_pred = []
        for inputs in dataloader:
            y_pred.append(self.network(**inputs).detach())
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.numpy()
        return y_pred

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch regressors.

        PyTorch regressors require 3D input and we don't allow automatic conversion
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch regressors require properly formatted "
                f"3D time series data. Please reshape your data or use a supported Panel mtype."
            )

        # Call parent method for other conversions
        return super()._internal_convert(X, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for regressors:
                "results_comparison" - used for identity testing in some regressors
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return []


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning regressor based on pytorch."""

    def __init__(self, X, y=None):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        x = self.X[i]
        x = torch.tensor(x, dtype=torch.float)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torch.tensor(y, dtype=torch.float)
        return inputs, y
