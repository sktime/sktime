"""Exponential Smoothing Recurrant Neural Network (ES-RNN)."""

__author__ = ["Ankit-1204"]

from warnings import warn

from sktime.networks.base import BaseDeepNetwork
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


import numpy as np


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

    class _ESRNN(nn_module):
        def __init__(
            self,
            input_shape,
            hidden_size,
            horizon,
            num_layer,
            level_coeff,
            seasonal_coeff_1,
            season_length,
            seasonality="zero",
        ) -> None:
            self.input_shape = input_shape
            self.hidden_size = hidden_size
            self.num_layer = num_layer
            self.horizon = horizon
            self.seasonality = seasonality
            self.level_coeff = level_coeff
            self.seasonal_coeff_1 = seasonal_coeff_1
            self.season_length = season_length
            super().__init__()
            self.input_layer = torch.Sequential(nn.Linear(input_shape))
            self.output_layer = torch.Sequential(
                nn.LSTM(self.input_shape, self.hidden_size, self.num_layer),
                nn.Linear(self.hidden_size, self.horizon),
            )

        def _nonseasonal(self, x):
            """Calculate and returns the level at last data point."""
            level = x[0]
            for t in range(1, len(x)):
                level = self.level_coeff * t + (1 - self.level_coeff) * level

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
            self.level_coeff = nn.Sigmoid(self.level_coeff)
            self.seasonal_coeff_1 = nn.Sigmoid(self.seasonal_coeff_1)
            for i in range(len(x)):
                season_index = i % season_length
                new_level = (
                    self.level_coeff * (x[i] / seasonality[season_index])
                    + (1 - self.level_coeff) * level
                )
                seasonality[season_index] = (
                    self.seasonal_coeff_1 * (x[i] / new_level)
                    + (1 - self.seasonal_coeff_1) * seasonality[season_index]
                )
                level = new_level
            return (
                level,
                seasonality,
                x / (level * seasonality[torch.arange(length) % season_length]),
            )

        def _double_seasonal(level_coeff, seasonal_coeff_1, x):
            pass

        def forward(self, x):
            """
            Forward pass through ES-RNN.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_length).
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype("float32"))
            self._seasonality = self.seasonality
            self._horizon = self.horizon
            if self._seasonality == "zero":
                level, new_x = self._nonseasonal(x)

                x_input = self.input_layer(new_x.float())
                output = self.output_layer(x_input.unsqueeze(0))

                output_leveled = output * level
                return output_leveled
            elif self.seasonality == "single":
                level, seasonality, new_x = self._single_seasonal(
                    x, self.level_coeff, self.seasonal_coeff_1, self.season_length
                )
                x_input = self.input_layer(new_x.float())
                output = self.output_layer(x_input.unsqueeze(0))

                output_leveled = (
                    output
                    * level
                    * seasonality[torch.arange(self._horizon) % self.season_length]
                )
                return output_leveled

    def __init__(
        self,
        input_shape=1,
        hidden_size=1,
        horizon=1,
        num_layer=1,
        level_coeff=0.5,
        seasonal_coeff_1=0.5,
        season_length=12,
        seasonality="zero",
    ) -> None:
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.horizon = horizon
        self.level_coeff = level_coeff
        self.seasonal_coeff_1 = seasonal_coeff_1
        self.season_length = season_length
        self.seasonality = seasonality
        super().__init__()

    def build_network(self):
        """Build the ES-RNN."""
        return self._ESRNN(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.seasonality,
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
            "input_shape": 3,
            "hidden_size": 3,
            "horizon": 5,
            "num_layer": 5,
            "level_coeff": 0.5,
            "seasonal_coeff_1": 0.5,
            "season_length": 12,
            "seasonality": "zero",
        }
        return [params1, params2]
