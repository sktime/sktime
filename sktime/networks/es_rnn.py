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
        seasonality="zero",
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
                seasonality="zero",
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
                self.output_layer = nn.Linear(self.hidden_size, self.horizon)

            def _nonseasonal(self, x):
                """Calculate and returns the level at last data point."""
                level = x[0]
                level_coeff = torch.sigmoid(self.level_coeff)
                for t in range(1, len(x)):
                    level = level_coeff * t + (1 - level_coeff) * level

                return level, x / level

            def _single_seasonal(self, x):
                length = len(x)
                if self.season_length > length:
                    warn("Input window should atleast cover one season")
                    season_length = length
                n_season = length // season_length

                avg_per_season = [
                    torch.mean(x[i * season_length : (i + 1) * season_length])
                    for i in range(n_season)
                ]
                level = torch.mean(x[:season_length])
                seasonality = torch.array(
                    [
                        x[i] / avg_per_season[i // season_length]
                        for i in range(season_length)
                    ]
                )
                seasonality = torch.exp(seasonality)
                level_coeff = torch.sigmoid(self.level_coeff)
                seasonal_coeff_1 = torch.sigmoid(self.seasonal_coeff_1)
                for i in range(len(x)):
                    season_index = i % season_length
                    new_level = (
                        level_coeff * (x[i] / seasonality[season_index])
                        + (1 - level_coeff) * level
                    )
                    seasonality[season_index] = (
                        seasonal_coeff_1 * (x[i] / new_level)
                        + (1 - seasonal_coeff_1) * seasonality[season_index]
                    )
                    level = new_level
                return (
                    level,
                    seasonality,
                    x / (level * seasonality[torch.arange(length) % season_length]),
                )

            def _double_seasonal(x):
                pass

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
                    output = self.output_layer(output)
                    output_leveled = output * level
                    return output_leveled
                elif self.seasonality == "single":
                    level, seasonality, new_x = self._single_seasonal(
                        x, self.level_coeff, self.seasonal_coeff_1, self.season_length
                    )
                    x_input = self.input_layer(new_x.float())
                    output, _ = self.lstm(x_input)
                    output = self.output_layer(output)

                    output_leveled = (
                        output
                        * level
                        * seasonality[torch.arange(self._horizon) % self.season_length]
                    )
                else:
                    pass
                    return output_leveled

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
