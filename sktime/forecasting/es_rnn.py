"""Interface for ES RNN for Time Series Forecasting."""

import numpy as np

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.es_rnn import ESRNN
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class ESRNNTrainDataset(Dataset):
    """Implements Pytorch Dataset class for Training ESS-RNN."""

    def __init__(self, y, pred_len, window, stride) -> None:
        self.y = y
        self.window = window
        self.pred_len = pred_len
        self.stride = stride
        self._get_data()

    def _get_data(self):
        length = len(self.y)
        x_arr = []
        y_arr = []
        for i in range(0, length - self.window - self.pred_len + 1, self.stride):
            inp = self.y[i : i + self.window]
            out = self.y[i + self.window : i + self.window + self.pred_len]

            x_arr.append(inp)
            y_arr.append(out)

        if not x_arr:
            raise ValueError("Input size to small")

        self.x_train, self.y_train = (
            torch.FloatTensor(np.array(x_arr)),
            torch.FloatTensor(np.array(y_arr)),
        )

    def __len__(self):
        """Get length of the dataset."""
        return len(self.x_train)

    def __getitem__(self, idx):
        """Get data pairs at this index."""
        return self.x_train[idx], self.y_train[idx]


class ESRNNPredDataset(Dataset):
    """Implements Pytorch Dataset class for Prediction ESS-RNN."""

    def __init__(self, y, window) -> None:
        self.y = y
        self.window = window
        self._get_data()

    def _get_data(self):
        x_pred = self.y[-self.window :]
        x_pred = torch.FloatTensor(np.array(x_pred))
        self.x_pred = x_pred.unsqueeze(0)

    def __len__(self):
        """Get length of the dataset."""
        return len(self.x_pred)

    def __getitem__(self, idx):
        """Return data point."""
        return self.x_pred[idx], torch.zeros(1)


class ESRNNForecaster(BaseDeepNetworkPyTorch):
    """
    Exponential Smoothing Recurrant Neural Network.

    This model combines Exponential Smoothing (ES) and (LSTM) networks
    for time series forecasting. ES is used to balance the level and
    seasonality of the series. This method has been proposed in [1]_.

    Parameters
    ----------
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
    stride : int
        stride for sliding window
    batch_size : int
        size of batch during training
    num_epochs : int
        number of epochs during training
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
    lr : int
        Learning rate for training

    References
    ----------
    .. [1] Smyl, S. 2020.
    A hybrid method of exponential smoothing and recurrent \
    neural networks for time series forecasting.
    https://www.sciencedirect.com/science/article/pii/S0169207019301153

    Examples
    --------
    >>> from sktime.forecasting.es_rnn import ESRNNForecaster # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.boxcox import LogTransformer
    >>> y = load_airline()
    >>> scaler=LogTransformer()
    >>> forecaster=ESRNNForecaster(15,6,12,6,'double',20,1,32,100,'MSE')# doctest: +SKIP
    >>> y_new=scaler.fit_transform(y)
    >>> forecaster.fit(y_new, fh=[1,2,3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP
    >>> y_pred=scaler.inverse_transform(y_pred) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ankit-1204"],
    }

    def __init__(
        self,
        hidden_size=10,
        num_layer=5,
        season1_length=12,
        season2_length=6,
        seasonality_type="single",
        window=10,
        stride=1,
        batch_size=32,
        num_epochs=1000,
        criterion=None,
        optimizer="Adam",
        lr=1e-1,
        optimizer_kwargs=None,
        criterion_kwargs=None,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ) -> None:
        super().__init__()
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
        self.custom_dataset_pred = custom_dataset_pred
        self.lr = lr
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

    def _instantiate_criterion(self):
        if self.criterion:
            return super()._instantiate_criterion()
        else:
            return ESRNN().pin_ball()

    def _build_network(self, fh):
        self.pred_len = fh
        self.input_shape = self._y.shape[-1]
        return ESRNN(
            self.input_shape,
            self.hidden_size,
            self.pred_len,
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
            dataset = ESRNNTrainDataset(
                y=y,
                window=self.window,
                pred_len=self.pred_len,
                stride=self.stride,
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
            dataset = ESRNNPredDataset(
                y=y,
                window=self.window,
            )
        return DataLoader(
            dataset,
            self.batch_size,
        )

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
            "hidden_size": 10,
            "num_layer": 5,
            "season1_length": 2,
            "season2_length": 2,
            "seasonality_type": "single",
            "window": 3,
            "stride": 1,
            "batch_size": 32,
            "num_epochs": 10,
            "custom_dataset_train": None,
            "custom_dataset_pred": None,
            "optimizer": "Adam",
            "lr": 1e-3,
        }
        return [params1, params2]
