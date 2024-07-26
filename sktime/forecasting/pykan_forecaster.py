# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy forecasters."""

__author__ = ["bheidri"]

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


if _check_soft_dependencies("kan", severity="none"):
    from kan import KAN


class PyKANForecaster(BaseForecaster):
    """
    PyKANForecaster uses Kolmogorov Arnold Network [1] to forecast time series data.

    This forecaster uses the pykan library to create a KAN model to forecast
    time series data. The model is trained on the training data and then used to
    forecast the future values of the time series.
    **Note** This forecaster is experimental and the used library for implementing
    the KANs may be exchanged in the future to a more stable and efficient library.

    Parameters
    ----------
    hidden_layers : tuple, optional (default=(1, 1))
        The number of hidden layers in the network.
    input_layer_size : int, optional (default=2)
        The size of the input layer.
    k : int, optional (default=3)
        The number of nearest neighbors to consider.
    grids : np.array, optional (default=np.array([2, 3]))
        The grid sizes to use in the model.
    model_params : dict, optional (default={"k": 2})
        The parameters to pass to the model. See pykan documentation for more details.
    fit_params : dict, optional (default={"steps": 1})
        The parameters to pass to the fit method. See pykan documentation
        for more details.
    val_size : float, optional (default=0.5)
        The size of the validation set to use in the training.
    device : str, optional (default="cpu")
        The device to use for training the model.

    References
    ----------
    .. [1] Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
      arXiv preprint arXiv:2404.19756 (2024).
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:insample": False,
        "python_version": None,
        "python_dependencies": ["pykan", "torch"],
        "python_dependencies_alias": {"pykan": "kan"},
    }

    def __init__(
        self,
        hidden_layers=(1, 1),
        input_layer_size=2,
        grids=None,
        model_params=None,
        fit_params=None,
        val_size=0.5,
        device="cpu",
    ):
        self.hidden_layers = hidden_layers
        self.input_layer_size = input_layer_size
        self.grids = grids
        self._grids = grids if grids is not None else [2, 3]
        self.val_size = val_size
        self.model_params = model_params
        self._model_params = model_params if model_params is not None else {"k": 2}
        self.fit_params = fit_params
        self._fit_params = fit_params if fit_params is not None else {"steps": 1}
        self.device = device
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
        output_size = max(fh.to_relative(self.cutoff)._values)
        if X is not None:
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X=X, test_size=(self.val_size)
            )
            ds_train = PyTorchTrainDataset(
                y_train, self.input_layer_size, output_size, X=X_train
            )
            ds_test = PyTorchTrainDataset(
                y_test, self.input_layer_size, output_size, X=X_test
            )
            input_layer_size = self.input_layer_size + X_train.shape[1] * output_size
        else:
            y_train, y_test = temporal_train_test_split(y, test_size=(self.val_size))
            ds_train = PyTorchTrainDataset(y_train, self.input_layer_size, output_size)
            ds_test = PyTorchTrainDataset(y_test, self.input_layer_size, output_size)
            input_layer_size = self.input_layer_size
        train_input = [ds_train[i][0] for i in range(len(ds_train))]
        train_target = [ds_train[i][1] for i in range(len(ds_train))]
        test_input = [ds_test[i][0] for i in range(len(ds_test))]
        test_target = [ds_test[i][1] for i in range(len(ds_test))]
        # no fitting, we already know the forecast values

        ds_new = {
            "train_input": torch.stack(train_input).type(torch.float32).to(self.device),
            "train_label": torch.stack(train_target)
            .type(torch.float32)
            .to(self.device),
            "test_input": torch.stack(test_input).type(torch.float32).to(self.device),
            "test_label": torch.stack(test_target).type(torch.float32).to(self.device),
        }

        self.train_losses = []
        self.test_losses = []

        self._layer_sizes = [input_layer_size, *self.hidden_layers, output_size]
        for i in range(len(self._grids)):
            if i == 0:
                model = KAN(
                    width=self._layer_sizes,
                    grid=self._grids[i],
                    device=self.device,
                    **self._model_params,
                )
            if i != 0:
                model = KAN(
                    width=self._layer_sizes,
                    grid=self._grids[i],
                    device=self.device,
                    **self._model_params,
                ).initialize_from_another_model(model, ds_new["train_input"])
            results = model.fit(ds_new, device=self.device, **self._fit_params)
            if len(self.test_losses) == 0 or results["test_loss"][-1] < min(
                self.test_losses
            ):
                self._state_dict = model.state_dict()
                self._best_grid = self._grids[i]
            self.train_losses += results["train_loss"]
            self.test_losses += results["test_loss"]
        # self.state_dict = best_model.state_dict()
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
        model = KAN(width=self._layer_sizes, grid=self._best_grid, **self._model_params)
        model.load_state_dict(self._state_dict)
        if X is not None:
            input_ = torch.cat(
                [
                    torch.from_numpy(self._y.values[-self.input_layer_size :]).reshape(
                        (1, -1)
                    ),
                    torch.from_numpy(X.values).reshape((1, -1)),
                ],
                dim=-1,
            ).type(torch.float32)
        else:
            input_ = (
                torch.from_numpy(self._y.values[-self.input_layer_size :])
                .reshape((1, -1))
                .type(torch.float32)
            )

        prediction = model(input_).detach().numpy().reshape((-1,))
        index = list(fh.to_absolute(self.cutoff))
        return pd.Series(
            prediction[fh.to_relative(self._cutoff) - 1], index=index, name=self._y.name
        )

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
            {},  # default parameters
            {
                "grids": [2, 3],
                "model_params": {"k": 2},
                "fit_params": {"steps": 1},
                "input_layer_size": 2,
                "hidden_layers": (1,),
            },
            {
                "input_layer_size": 2,
                "grids": [3],
                "model_params": {"k": 2},
                "fit_params": {"steps": 1},
                "hidden_layers": (1, 1),
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

        hist_y = tensor(self.y[i : i + self.seq_len]).type(torch.float32)
        if self.X is not None:
            exog_data = (
                tensor(self.X[i + self.seq_len : i + self.seq_len + self.fh])
                .type(torch.float32)
                .flatten()
            )
        else:
            exog_data = tensor([])
        return (
            torch.cat([hist_y, exog_data]),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).type(
                torch.float32
            ),
        )
