# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy forecasters."""

__author__ = ["bheidri"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""

        pass


if _check_soft_dependencies("kan", severity="none"):
    from kan import KAN


class PyKANForecaster(BaseForecaster):
    """
    PyKANForecaster uses Kolmogorov Arnold Network [1] to forecast time series data.

    This forecaster uses the pykan library to create a KAN model to forecast
    time series data. The model is trained on the training data and then used to
    forecast the future values of the time series.

    Parameters
    ----------
    hidden_layers : tuple, optional (default=(5,5))
        The number of hidden layers in the network.
    input_layer_size : int, optional (default=12)
        The size of the input layer.
    k : int, optional (default=3)
        The number of nearest neighbors to consider.
    grids : np.array, optional (default=np.array([5,10,20, 50, 100]))
        The grid sizes to use in the model.
    stop_grid_update_step : int, optional (default=30)
        The number of steps to wait before updating the grid.
    opt : str, optional (default="LBFGS")
        The optimization algorithm to use.
    steps : int, optional (default=5)
        The number of steps to train the model.


    References
    ----------
    .. [1] Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
      arXiv preprint arXiv:2404.19756 (2024).
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_version": None,
        "python_dependencies": ["kan", "torch"],
    }

    def __init__(
        self,
        hidden_layers=(5, 5),
        input_layer_size=12,
        k=3,
        grids=None,
        stop_grid_update_step=30,
        opt="LBFGS",
        steps=5,
    ):
        self.hidden_layers = hidden_layers
        self.input_layer_size = input_layer_size
        self.k = k
        self.grids = grids
        self._grids = grids if grids is not None else [5, 10, 20, 50, 100]
        self.stop_grid_update_step = stop_grid_update_step
        self.opt = opt
        self.steps = steps
        super().__init__()

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        output_size = max(fh._values)
        y_train, y_test = temporal_train_test_split(y, test_size=36)
        ds = PyTorchTrainDataset(y_train, self.input_layer_size, output_size)
        train_y = [ds[i][0] for i in range(len(ds))]
        train_target = [ds[i][1] for i in range(len(ds))]

        ds = PyTorchTrainDataset(y_test, self.input_layer_size, output_size)
        test_y = [ds[i][0] for i in range(len(ds))]
        test_target = [ds[i][1] for i in range(len(ds))]
        # no fitting, we already know the forecast values

        ds_new = {
            "train_input": torch.stack(train_y),
            "train_label": torch.stack(train_target),
            "test_input": torch.stack(test_y),
            "test_label": torch.stack(test_target),
        }

        self.train_losses = []
        self.test_losses = []

        best_model = None
        for i in range(len(self._grids)):
            if i == 0:
                model = KAN(
                    width=[self.input_layer_size, *self.hidden_layers, output_size],
                    grid=self.grids[i],
                    k=self.k,
                )
            if i != 0:
                model = KAN(
                    width=[self.input_layer_size, *self.hidden_layers, output_size],
                    grid=self._grids[i],
                    k=self.k,
                ).initialize_from_another_model(model, ds_new["train_input"])
            results = model.train(
                ds_new,
                opt=self.opt,
                steps=self.steps,
                stop_grid_update_step=self.stop_grid_update_step,
            )
            if len(self.test_losses) == 0 or results["test_loss"][-1] < min(
                self.test_losses
            ):
                best_model = model
            self.train_losses += results["train_loss"]
            self.test_losses += results["test_loss"]
        self.model = best_model
        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        input_ = torch.from_numpy(self._y.values[-self.input_layer_size :]).reshape(
            (1, -1)
        )
        prediction = self.model(input_).detach().numpy().reshape((-1,))

        index = list(fh.to_absolute(self.cutoff))
        return pd.Series(prediction, index=index)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {
                "steps": 1,
                "grids": [10, 20],
                "k": 2,
            },
            {
                "steps": 1,
                "grids": [10],
                "k": 3,
                "hidden_layers": (2, 2),
            },
        ]
        return params


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
