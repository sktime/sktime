"""Interface for ES RNN for Time Series Forecasting."""

__author__ = ["Ankit-1204"]
import numpy as np

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.es_rnn import ESRNN
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn:
        """dummy class if torch is not available."""

        class Module:
            """dummy class if torch is not available."""

            def __init__(self, *args, **kwargs):
                raise ImportError("torch is not available. Please install torch first.")


class ESRNNForecaster(BaseDeepNetworkPyTorch):
    """Exponential Smoothing Recurrant Neural Network."""

    def __init__(
        self,
        input_shape,
        hidden_size,
        num_layer,
        season_length,
        seasonality="zero",
        window=5,
        stride=1,
        batch_size=32,
        epoch=50,
    ) -> None:
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.seasonality = seasonality
        self.level_coeff = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.seasonal_coeff_1 = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.season_length = season_length
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.epoch = epoch
        super().__init__()

    def _get_windows(self, y):
        length = len(y)
        x_arr = [], y_arr = []
        for i in range(0, length - self.window - self.horizon + 1, self.stride):
            inp = y[i : i + self.window]
            out = y[i + self.window : i + self.window + self.horizon]

            x_arr.append(inp.flatten())
            y_arr.append(out.flattent())

        if not x_arr:
            raise ValueError("Input size to small")

        return np.array(x_arr), np.array(y_arr)

    def _fit(self, y, fh, X=None):
        """Fit ES-RNN Model for provided data."""
        from torch.utils.data import DataLoader, TensorDataset

        self.horizon = fh
        self.model = ESRNN(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.level_coeff,
            self.seasonal_coeff_1,
            self.season_length,
            self.seasonality,
        ).build_network()
        x_train, y_train = self._get_windows(y)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        data = TensorDataset(x_train, y_train)
        loader = DataLoader(data, self.batch_size, shuffle=True)
        self.model.train()
        for i in range(self.epoch):
            self._run_epoch(i, loader)
