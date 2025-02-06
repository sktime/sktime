"""Exponential Smoothing Recurrant Neural Network (ES-RNN)."""

__author__ = ["Ankit-1204"]

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

import numpy as np


def _nonseasonal(x, alpha=0.3):
    """Calculate and returns the level at last data point."""
    level = x[0]
    for t in range(1, len(x)):
        level = alpha * t + (1 - alpha) * level

    return level, x / level


def _single_seasonal(alpha, beta, season_length, x):
    length = len(x)
    if season_length > length:
        season_length = length
    n_season = length // season_length
    avg_per_season = [
        np.mean(x[i * season_length : (i + 1) * season_length]) for i in range(n_season)
    ]
    level = np.mean(x[:season_length])
    seasonality = np.array(
        [x[i] / avg_per_season[i // season_length] for i in range(season_length)]
    )
    seasonality = np.exp(seasonality)
    alpha = nn.Sigmoid(alpha)
    beta = nn.Sigmoid(beta)
    for i in range(len(x)):
        season_index = i % season_length
        new_level = alpha * (x[i] / seasonality[season_index]) + (1 - alpha) * level
        seasonality[season_index] = (
            beta * (x[i] / new_level) + (1 - beta) * seasonality[season_index]
        )
        level = new_level

    return (
        level,
        seasonality,
        x / (level * seasonality[np.arange(length) % season_length]),
    )


def _double_seasonal(alpha, beta, x):
    pass


class ESRNN:
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

    alpha : int

    beta : int

    """

    class _ESRNN(nn.Module):
        def __init__(
            self,
            input_shape,
            hidden_size,
            horizon,
            num_layer,
            alpha,
            beta,
            season_length,
            seasonality="zero",
        ) -> None:
            self.input_shape = input_shape
            self.hidden_size = hidden_size
            self.num_layer = num_layer
            self.horizon = horizon
            self.seasonality = seasonality
            self.alpha = alpha
            self.beta = beta
            self.season_length = season_length
            super().__init__()
            self.input_layer = torch.Sequential(nn.Linear(input_shape))
            self.output_layer = torch.Sequential(
                nn.LSTM(self.input_shape, self.hidden_size, self.num_layer),
                nn.Linear(self.hidden_size, self.horizon),
            )

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
                level, new_x = _nonseasonal(x)

                x_input = self.input_layer(new_x.float())
                output = self.output_layer(x_input.unsqueeze(0))

                output_leveled = output * level
                return output_leveled
            elif self.seasonality == "single":
                level, seasonality, new_x = _single_seasonal(
                    x, self.alpha, self.beta, self.season_length
                )
                x_input = self.input_layer(new_x.float())
                output = self.output_layer(x_input.unsqueeze(0))

                output_leveled = (
                    output
                    * level
                    * seasonality[np.arange(self._horizon) % self.season_length]
                )
                return output_leveled

    def __init__(
        self, input_shape, hidden_size, horizon, num_layer, seasonality="zero"
    ) -> None:
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.horizon = horizon
        self.seasonality = seasonality

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
            "alpha": 0.5,
            "beta": 0.5,
            "season_length": 12,
            "seasonality": "zero",
        }
        return [params1, params2]
