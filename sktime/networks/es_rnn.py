"""Exponential Smoothing Recurrant Neural Network (ES-RNN)."""

__author__ = ["Ankit-1204"]

from warnings import warn

from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _check_soft_dependencies


class ESRNN(BaseDeepNetwork):
    """
    Exponential Smoothing Recurrant Neural Network.

    This model combines Exponential Smoothing (ES) and (LSTM) networks
    for time series forecasting. ES is used to balance the level and
    seasonality of the series.

    References
    ----------
    [1] Smyl, S. (2020). A hybrid method of exponential smoothing and
        recurrent neural networks for time series forecasting.

        https://www.sciencedirect.com/science/article/pii/S0169207019301153

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
        Coefficient for smoothing of the level component

    seasonal_coeff_1 : int
        Coefficient for smoothing of the season component

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
        horizon=1,
        num_layer=1,
        season_length=12,
        seasonality="single",
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.horizon = horizon
        self.season_length = season_length
        self.seasonality = seasonality

        if _check_soft_dependencies("torch", severity="none"):
            import torch
            import torch.nn as nn

            NNModule = nn.Module
        else:

            class NNModule:
                """Dummy class if torch is unavailable."""

        class PinballLoss(NNModule):
            """
            Default Pinball/Quantile Loss.

            Parameters
            ----------
            tau : Quantile Value
            target : Ground truth
            predec: Predicted value
            loss = max( (predec-target)(1-tau), (target-predec)*tau)
            """

            def __init__(self, tau=0.49):
                super().__init__()
                self.tau = tau

            def forward(self, predec, target):
                """Calculate Pinball Loss."""
                predec = predec.float()
                target = target.float()
                diff = predec - target
                loss = torch.maximum(-diff * (1 - self.tau), diff * self.tau)
                return loss.mean()

        class _ESRNN(NNModule):
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
                """
                Calculate and returns the level and seasonality.

                Used for single seasonality condition.

                """
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
        self.loss = PinballLoss

    def DefaultLoss(self):
        """Return the default Pinball Loss."""
        return self.loss()

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
            "horizon": 1,
            "num_layer": 1,
            "season_length": 3,
            "seasonality": "single",
        }
        return [params1, params2]
