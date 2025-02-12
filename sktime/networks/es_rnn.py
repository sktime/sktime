"""Exponential Smoothing Recurrant Neural Network (ES-RNN)."""

__author__ = ["Ankit-1204"]

from warnings import warn

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_soft_dependencies


class ESRNN(BaseDeepNetwork):
    """
    Exponential Smoothing Recurrant Neural Network.

    Parameters
    ----------
    input_shape : int
        Number of features in the input

    hidden_size : int
        Number of features in the hidden state

    horizon : int
        Forecasting horizon

    num_layer : int
        Number of layers

    season_length : int
        Period of season

    seasonality : string
        Type of seasonality

    level_coeff : int

    seasonal_coeff_1 : int

    """

    def __init__(
        self,
        input_shape=1,
        hidden_size=1,
        horizon=1,
        num_layer=1,
        season_length=12,
        seasonality="single",
    ) -> None:
        super().__init__()
        _check_soft_dependencies("torch", severity="none")
        import torch
        import torch.nn as nn

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.horizon = horizon
        self.season_length = season_length
        self.seasonality = seasonality

        class _ESRNN(nn.Module):
            def __init__(
                self,
                input_shape,
                hidden_size,
                horizon,
                num_layer,
                season_length,
                seasonality,
            ) -> None:
                self.input_shape = input_shape
                self.hidden_size = hidden_size
                self.num_layer = num_layer
                self.horizon = horizon
                self.seasonality = seasonality
                self.season_length = season_length
                super().__init__()
                self.level_coeff = torch.nn.Parameter(torch.rand(1), requires_grad=True)
                self.seasonal_coeff_1 = torch.nn.Parameter(
                    torch.rand(1), requires_grad=True
                )
                self.input_layer = nn.Linear(input_shape, input_shape)
                self.lstm = nn.LSTM(
                    self.input_shape, self.hidden_size, self.num_layer, batch_first=True
                )
                self.output_layer = nn.Linear(self.hidden_size, self.input_shape)

            def _nonseasonal(self, x):
                """Calculate and returns the level at last data point."""
                batch, seq_length, num_features = x.shape
                level = x[:, 0, :]
                level_coeff = torch.sigmoid(self.level_coeff)
                for t in range(1, seq_length):
                    level = level_coeff * x[:, t, :] + (1 - level_coeff) * level

                return level, x / level.unsqueeze(1)

            def _single_seasonal(self, x):
                batch, seq_length, num_features = x.shape
                season_length = self.season_length
                if self.season_length > seq_length:
                    warn(f"Input window should atleast cover one season,{seq_length}")
                    season_length = seq_length
                level = x[:, :season_length, :].mean(dim=1, keepdim=True)
                initial_seasonality = x[:, :season_length, :] / level
                seasonality = []
                for i in range(season_length):
                    seasonality.append(torch.exp(initial_seasonality[:, i, :]))
                level_coeff = torch.sigmoid(self.level_coeff)
                seasonal_coeff_1 = torch.sigmoid(self.seasonal_coeff_1)
                for i in range(seq_length):
                    new_level = level_coeff * (x[:, i, :] / seasonality[i]) + (
                        1 - level_coeff
                    ) * level.squeeze(1)

                    seasonality.append(
                        seasonal_coeff_1 * (x[:, i, :] / new_level)
                        + (1 - seasonal_coeff_1) * seasonality[i]
                    )
                    level = new_level.unsqueeze(1)
                seasonality = torch.stack(seasonality, dim=1)
                return (
                    level,
                    seasonality,
                    x / (level * seasonality[:, -seq_length:, :]),
                )

            def forward(self, x):
                """
                Forward pass through ES-RNN.

                Parameters
                ----------
                x : torch.Tensor
                    Input tensor of shape (batch_size, input_length).
                """
                if self.seasonality == "zero":
                    level, new_x = self._nonseasonal(x)
                    x_input = self.input_layer(new_x.float())
                    output, _ = self.lstm(x_input)
                    output = self.output_layer(output[:, -self.horizon :, :])
                    output_leveled = (output) * level.unsqueeze(-1)
                    return output_leveled

                elif self.seasonality == "single":
                    level, seasonality, new_x = self._single_seasonal(x)
                    x_input = self.input_layer(new_x.float())
                    output, _ = self.lstm(x_input)
                    output = self.output_layer(output[:, -self.horizon :, :])
                    output_leveled = (
                        (output) * level * seasonality[:, -self.horizon :, :]
                    )
                    return output_leveled
                else:
                    pass

        self._network_class = _ESRNN

    def build_network(self):
        """Build the ES-RNN."""
        return self._network_class(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.season_length,
            self.seasonality,
        )
