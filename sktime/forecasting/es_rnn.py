"""Interface for ES RNN for Time Series Forecasting."""

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.es_rnn import ESRNN
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class ESRNNDataset(Dataset):
    """Implements Pytorch Dataset class for ESS-RNN."""

    def __init__(self, y, horizon, window, stride) -> None:
        self.y = y
        self.window = window
        self.horizon = horizon
        self.stride = stride
        x_train, y_train = self._get_windows(y)
        self.x_train = torch.FloatTensor(x_train)
        self.y_train = torch.FloatTensor(y_train)

    def _get_windows(self, y):
        length = len(y)
        x_arr = []
        y_arr = []
        for i in range(0, length - self.window - self.horizon + 1, self.stride):
            inp = y[i : i + self.window]
            out = y[i + self.window : i + self.window + self.horizon]

            x_arr.append(inp)
            y_arr.append(out)

        if not x_arr:
            raise ValueError("Input size to small")

        return np.array(x_arr), np.array(y_arr)

    def __len__(self):
        """Get length of the dataset."""
        return len(self.x_train)

    def __getitem__(self, idx):
        """Get data pairs at this index."""
        return self.x_train[idx], self.y_train[idx]


class ESRNNForecaster(BaseDeepNetworkPyTorch):
    """
    Exponential Smoothing Recurrant Neural Network.

    This model combines Exponential Smoothing (ES) and (LSTM) networks
    for time series forecasting. ES is used to balance the level and
    seasonality of the series. This method has been proposed in [1]_.

    Parameters
    ----------
    input_shape : int
        Number of features in the input
    hidden_size : int
        Number of features in the hidden state
    num_layer : int
        Number of layers
    seasonality_type : string
        Type of seasonality_type, could be zero ,single or double
    season1_length : int
        Period of season 1
    season2_length : int
        Period of season 2
    criterion : torch.nn Loss Function, default=torch.nn.MSELoss
        loss function to be used for training
    criterion_kwargs : dict, default=None
        keyword arguments to pass to criterion
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        optimizer to be used for training
    optimizer_kwargs : dict, default=None
        keyword arguments to pass to optimizer
    window : int
        Size of Input window, default=5

    References
    ----------
    .. [1] Smyl, S. 2020.
    A hybrid method of exponential smoothing and recurrent \
    neural networks for time series forecasting.
    https://www.sciencedirect.com/science/article/pii/S0169207019301153

    Examples
    --------
    >>> from sktime.forecasting.es_rnn import ESRNNForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = ESRNNForecaster(10, 3)
    >>> forecaster.fit(y, fh=[1,2,3])
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ankit-1204"],
    }

    def __init__(
        self,
        input_shape=1,
        hidden_size=1,
        num_layer=1,
        season1_length=3,
        season2_length=3,
        seasonality_type="zero",
        window=5,
        stride=1,
        batch_size=32,
        num_epochs=10,
        custom_dataset_train=None,
        optimizer="Adam",
        optimizer_kwargs=None,
        criterion="pinball",
        criterion_kwargs=None,
        lr_rate=1e-5,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.seasonality_type = seasonality_type
        self.season1_length = season1_length
        self.season2_length = season2_length
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.custom_dataset_train = custom_dataset_train
        self.lr_rate = lr_rate
        if _check_soft_dependencies("torch", severity="none"):
            import torch

            self.criterions = {
                "MSE": torch.nn.MSELoss,
                "L1": torch.nn.L1Loss,
                "SmoothL1": torch.nn.SmoothL1Loss,
                "Huber": torch.nn.HuberLoss,
            }

            self.optimizers = {
                "Adadelta": torch.optim.Adadelta,
                "Adagrad": torch.optim.Adagrad,
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "SGD": torch.optim.SGD,
            }

    def _get_windows(self, y):
        length = len(y)
        x_arr = []
        y_arr = []
        for i in range(0, length - self.window - self.horizon + 1, self.stride):
            inp = y[i : i + self.window]
            out = y[i + self.window : i + self.window + self.horizon]

            x_arr.append(inp)
            y_arr.append(out)

        if not x_arr:
            raise ValueError("Input size to small")

        return np.array(x_arr), np.array(y_arr)

    def _instantiate_criterion(self):
        if self.criterion:
            return super()._instantiate_criterion()
        else:
            return ESRNN().pin_ball()

    def _build_network(self, fh):
        self.horizon = len(fh)
        self.input_shape = self._y.shape[1]
        self.network = ESRNN(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.season1_length,
            self.season2_length,
            self.seasonality_type,
        )._build()

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
            dataset = ESRNNDataset(
                y=y,
                window=self.window,
                horizon=self.horizon,
                stride=self.stride,
            )

        return DataLoader(dataset, self.batch_size, shuffle=True)

    def _predict(self, X=None, fh=None):
        """
        Predict with fitted model.

        Parameters
        ----------
        X:  Optional, If X is not provided then forecast
            is made on the fitted series.

        fh: Forecasting horizon,
            not used since, forecasting horizon at time of
            fitting is used (direct mode only)
        """
        import torch

        self.network.eval()
        if X is None:
            index = self._fh.to_absolute(self._y.index[-1]).to_numpy()
            column = (
                list(self._y.columns) if isinstance(self._y, pd.DataFrame) else None
            )
            input = self._y[-self.window :]
            input = torch.FloatTensor(np.array(input))
            input = input.unsqueeze(0)
        else:
            index = self._fh.to_absolute(X.index[-1]).to_numpy()
            column = list(X.columns) if isinstance(X, pd.DataFrame) else None
            input = torch.FloatTensor(np.array(X[-self.window :]))
            input = input.unsqueeze(0)
        with torch.no_grad():
            prediction = self.network(input)
            return pd.DataFrame(prediction.squeeze(0), index=index, columns=column)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "input_shape": 1,
            "hidden_size": 1,
            "num_layer": 1,
            "season1_length": 2,
            "season2_length": 2,
            "seasonality_type": "single",
            "window": 3,
            "stride": 1,
            "batch_size": 32,
            "num_epochs": 10,
            "optimizer": "Adam",
            "criterion": "mse",
            "lr_rate": 1e-3,
        }
        return [params1, params2]
