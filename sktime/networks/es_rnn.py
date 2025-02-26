"""Exponential Smoothing Recurrant Neural Network (ES-RNN)."""

__author__ = ["Ankit-1204"]

from warnings import warn

from sktime.utils.dependencies import _check_soft_dependencies


class ESRNN:
    """
    Exponential Smoothing Recurrant Neural Network.

    This model combines Exponential Smoothing (ES) and (LSTM) networks
    for time series forecasting. ES is used to balance the level and
    seasonality of the series.

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

    season1_length : int
        Period of season

    seasonality : string
        Type of seasonality

    level_coeff : int
        Coefficient for smoothing of the level component

    seasonal_coeff_1 : int
        Coefficient for smoothing of the season component

    References
    ----------
    .. [1] Smyl, S. 2020.
    A hybrid method of exponential smoothing and recurrent \
    neural networks for time series forecasting.
    https://www.sciencedirect.com/science/article/pii/S0169207019301153
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
        season1_length=12,
        season2_length=2,
        seasonality="single",
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.horizon = horizon
        self.season1_length = season1_length
        self.season2_length = season2_length
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
                season1_length,
                season2_length,
                seasonality,
            ) -> None:
                self.input_shape = input_shape
                self.hidden_size = hidden_size
                self.num_layer = num_layer
                self.horizon = horizon
                self.seasonality = seasonality
                self.season1_length = season1_length
                self.season2_length = season2_length
                super().__init__()
                self.level_coeff = torch.nn.Parameter(torch.rand(1), requires_grad=True)
                self.seasonal_coeff_1 = torch.nn.Parameter(
                    torch.rand(1), requires_grad=True
                )
                self.seasonal_coeff_2 = torch.nn.Parameter(
                    torch.rand(1), requires_grad=True
                )
                self.input_layer = nn.Linear(input_shape, input_shape)
                self.lstm = nn.LSTM(
                    self.input_shape, self.hidden_size, self.num_layer, batch_first=True
                )
                self.output_layer = nn.Linear(self.hidden_size, self.input_shape)

            def _nonseasonal(self, x):
                """
                Calculate level component for time series without seasonal patterns.

                Args:
                    x: Input tensor of shape (batch, seq_length, num_features)

                Returns
                -------
                    tuple: (level, remainder)
                """
                batch, seq_length, num_features = x.shape
                level = x[:, 0, :]
                level_coeff = torch.sigmoid(self.level_coeff)
                for t in range(1, seq_length):
                    level = level_coeff * x[:, t, :] + (1 - level_coeff) * level

                return level, x / level.unsqueeze(1)

            def _single_seasonal(self, x):
                """
                Calculate and return the level and one seasonality components.

                Used for single seasonality condition where the time series exhibits
                one distinct seasonal patterns (e.g., Yearly pattern).

                Args:
                    x: Input tensor of shape (batch, seq_length, num_features)

                Returns
                -------
                    tuple: (level, seasonality1, remainder)
                """
                batch, seq_length, num_features = x.shape
                season1_length = self.season1_length
                if self.season1_length > seq_length:
                    warn(f"Input window should atleast cover one season,{seq_length}")
                    season1_length = seq_length
                level = x[:, :season1_length, :].mean(dim=1, keepdim=True)
                initial_seasonality_1 = x[:, :season1_length, :] / level
                seasonality_1 = []
                for i in range(season1_length):
                    seasonality_1.append(torch.exp(initial_seasonality_1[:, i, :]))
                level_coeff = torch.sigmoid(self.level_coeff)
                seasonal_coeff_1 = torch.sigmoid(self.seasonal_coeff_1)
                for i in range(seq_length):
                    new_level = level_coeff * (x[:, i, :] / seasonality_1[i]) + (
                        1 - level_coeff
                    ) * level.squeeze(1)

                    seasonality_1.append(
                        seasonal_coeff_1 * (x[:, i, :] / new_level)
                        + (1 - seasonal_coeff_1) * seasonality_1[i]
                    )
                    level = new_level.unsqueeze(1)
                seasonality_1 = torch.stack(seasonality_1, dim=1)
                return (
                    level,
                    seasonality_1,
                    x / (level * seasonality_1[:, -seq_length:, :]),
                )

            def _double_seasonal(self, x):
                """
                Calculate and return the level and two seasonality components.

                Used for double seasonality condition where the time series exhibits
                two distinct seasonal patterns (e.g., daily and weekly patterns).

                Args:
                    x: Input tensor of shape (batch, seq_length, num_features)

                Returns
                -------
                    tuple: (level, seasonality1, seasonality2, remainder)
                """
                batch, seq_length, num_features = x.shape
                season1_length = self.season1_length
                if self.season1_length > seq_length:
                    season1_length = seq_length
                    warn(f"Input window should atleast cover one season,{seq_length}")

                season2_length = self.season2_length
                if self.season2_length > seq_length:
                    season2_length = seq_length
                    warn(f"Input window should atleast cover one season,{seq_length}")

                level = x[:, : max(season1_length, season2_length), :].mean(
                    dim=1, keepdim=True
                )
                initial_seasonality_1 = x[:, :season1_length, :] / level
                initial_seasonality_2 = x[:, :season2_length, :] / level
                seasonality_1 = []
                seasonality_2 = []
                for i in range(season1_length):
                    seasonality_1.append(torch.exp(initial_seasonality_1[:, i, :]))
                for i in range(season2_length):
                    seasonality_2.append(torch.exp(initial_seasonality_2[:, i, :]))

                level_coeff = torch.sigmoid(self.level_coeff)
                seasonal_coeff_1 = torch.sigmoid(self.seasonal_coeff_1)
                seasonal_coeff_2 = torch.sigmoid(self.seasonal_coeff_2)
                for i in range(seq_length):
                    new_level = level_coeff * (
                        x[:, i, :] / seasonality_1[i] * seasonality_2[i]
                    ) + (1 - level_coeff) * level.squeeze(1)

                    seasonality_1.append(
                        seasonal_coeff_1 * (x[:, i, :] / new_level * seasonality_2[i])
                        + (1 - seasonal_coeff_1) * seasonality_1[i]
                    )
                    seasonality_2.append(
                        seasonal_coeff_2 * (x[:, i, :] / new_level * seasonality_1[i])
                        + (1 - seasonal_coeff_2) * seasonality_2[i]
                    )
                    level = new_level.unsqueeze(1)
                seasonality_1 = torch.stack(seasonality_1, dim=1)
                seasonality_2 = torch.stack(seasonality_2, dim=1)
                return (
                    level,
                    seasonality_1,
                    seasonality_2,
                    x
                    / (
                        level
                        * seasonality_1[:, -seq_length:, :]
                        * seasonality_2[:, -seq_length:, :]
                    ),
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
                    output_leveled = (output) * level.unsqueeze(1)
                    return output_leveled

                elif self.seasonality == "single":
                    level, seasonality_1, new_x = self._single_seasonal(x)
                    x_input = self.input_layer(new_x.float())
                    output, _ = self.lstm(x_input)
                    output = self.output_layer(output[:, -self.horizon :, :])
                    output_leveled = (
                        (output) * level * seasonality_1[:, -self.horizon :, :]
                    )
                    return output_leveled
                else:
                    level, seasonality_1, seasonality_2, new_x = self._double_seasonal(
                        x
                    )
                    x_input = self.input_layer(new_x.float())
                    output, _ = self.lstm(x_input)
                    output = self.output_layer(output[:, -self.horizon :, :])
                    output_leveled = (
                        (output)
                        * level
                        * seasonality_1[:, -self.horizon :, :]
                        * seasonality_2[:, -self.horizon :, :]
                    )
                    return output_leveled

        self._network_class = _ESRNN
        self.loss = PinballLoss

    def pin_ball(self):
        """Return the default Pinball Loss."""
        return self.loss()

    def _build(self):
        """Build the ES-RNN."""
        return self._network_class(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.season1_length,
            self.season2_length,
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
            "seasonality": "single",
            "season1_length": 3,
            "season2_length": 2,
        }
        return [params1, params2]
