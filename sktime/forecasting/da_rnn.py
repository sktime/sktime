# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN).

This module implements the DA-RNN as described in:

    A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
    Yao Qin et al.
"""

__author__ = ["sanskarmodi8"]

import numpy as np

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies

# Handle PyTorch as a soft dependency
if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
else:
    torch = None

    class DummyDARNNModule:
        """Dummy class to raise an error when PyTorch is missing."""


class DARNNModule(nn.Module if torch else DummyDARNNModule):
    """PyTorch module implementing the DA-RNN model with dual-stage attention.

    This module consists of:
      - An encoder that uses input attention to select relevant exogenous
      features at each time step.
      - A decoder that uses temporal attention to select relevant encoder
      hidden states for forecasting.
    """

    def __init__(
        self,
        input_size,
        encoder_hidden_size,
        decoder_hidden_size,
        attention_dim,
    ):
        """
        Initialize the DA-RNN module.

        Parameters
        ----------
        input_size : int
            Number of exogenous (driving) features.
        encoder_hidden_size : int
            Hidden state size for the encoder LSTM.
        decoder_hidden_size : int
            Hidden state size for the decoder LSTM.
        attention_dim : int
            Dimension of the attention space.
        """
        super().__init__()
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_dim = attention_dim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder: LSTMCell processing exogenous inputs
        self.encoder_cell = nn.LSTMCell(input_size, encoder_hidden_size)

        # Input Attention Parameters
        self.We = nn.Linear(2 * encoder_hidden_size, attention_dim)
        self.Ue = nn.Linear(1, attention_dim, bias=False)
        self.v_e = nn.Parameter(torch.randn(attention_dim))

        # Decoder: LSTMCell processing target history
        self.decoder_cell = nn.LSTMCell(1, decoder_hidden_size)

        # Temporal Attention Parameters (decoder)
        self.Wd = nn.Linear(2 * decoder_hidden_size, attention_dim)
        self.Ud = nn.Linear(encoder_hidden_size, attention_dim)
        self.v_d = nn.Parameter(torch.randn(attention_dim))

        # Projection layer to combine previous target and context vector
        self.decoder_input_proj = nn.Linear(1 + encoder_hidden_size, 1)

        # Final output layer combining decoder state and context vector
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

    def forward(self, X, y_history):
        """
        Forward pass for DA-RNN.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, T, input_size)
            Exogenous input sequence.
        y_history : torch.Tensor, shape (batch, T-1, 1)
            Past target values for the decoder.

        Returns
        -------
        y_pred : torch.Tensor, shape (batch, 1)
            Predicted target value.
        """
        batch_size, T, _ = X.size()
        encoder_hidden_states = []

        # Initialize encoder LSTM state
        h_enc = torch.zeros(batch_size, self.encoder_hidden_size, device=X.device)
        c_enc = torch.zeros(batch_size, self.encoder_hidden_size, device=X.device)

        # Encoder with Input Attention
        for t in range(T):
            x_t = X[:, t, :]  # (batch, input_size)
            state_concat = torch.cat(
                [h_enc, c_enc], dim=1
            )  # (batch, 2*encoder_hidden_size)
            state_proj = self.We(state_concat)  # (batch, attention_dim)
            x_t_unsq = x_t.unsqueeze(-1)  # (batch, input_size, 1)
            feature_term = self.Ue(x_t_unsq)  # (batch, input_size, attention_dim)
            state_proj_exp = state_proj.unsqueeze(1).expand(
                -1, self.input_size, -1
            )  # (batch, input_size, attention_dim)
            e_t = torch.tanh(
                state_proj_exp + feature_term
            )  # (batch, input_size, attention_dim)
            e_t = torch.matmul(e_t, self.v_e)  # (batch, input_size)
            alpha_t = F.softmax(e_t, dim=1)  # (batch, input_size)
            x_t_tilde = alpha_t * x_t  # weighted input, (batch, input_size)
            h_enc, c_enc = self.encoder_cell(x_t_tilde, (h_enc, c_enc))
            encoder_hidden_states.append(h_enc.unsqueeze(1))
        encoder_hidden_states = torch.cat(
            encoder_hidden_states, dim=1
        )  # (batch, T, encoder_hidden_size)

        # Decoder with Temporal Attention
        decoder_steps = y_history.size(1)  # T-1 steps
        d_dec = torch.zeros(batch_size, self.decoder_hidden_size, device=X.device)
        c_dec = torch.zeros(batch_size, self.decoder_hidden_size, device=X.device)
        context_vector = None

        for t in range(decoder_steps):
            dec_state_concat = torch.cat(
                [d_dec, c_dec], dim=1
            )  # (batch, 2*decoder_hidden_size)
            dec_term = self.Wd(dec_state_concat)  # (batch, attention_dim)
            enc_term = self.Ud(encoder_hidden_states)  # (batch, T, attention_dim)
            dec_term_exp = dec_term.unsqueeze(1).expand(
                -1, T, -1
            )  # (batch, T, attention_dim)
            l_t = torch.tanh(dec_term_exp + enc_term)  # (batch, T, attention_dim)
            l_t = torch.matmul(l_t, self.v_d)  # (batch, T)
            beta_t = F.softmax(l_t, dim=1)  # (batch, T)
            context_vector = torch.sum(
                beta_t.unsqueeze(-1) * encoder_hidden_states, dim=1
            )  # (batch, encoder_hidden_size)
            y_prev = y_history[:, t, :]  # (batch, 1)
            dec_input = torch.cat(
                [y_prev, context_vector], dim=1
            )  # (batch, 1+encoder_hidden_size)
            dec_input_proj = self.decoder_input_proj(dec_input)  # (batch, 1)
            d_dec, c_dec = self.decoder_cell(dec_input_proj, (d_dec, c_dec))

        final_feature = torch.cat(
            [d_dec, context_vector], dim=1
        )  # (batch, decoder_hidden_size + encoder_hidden_size)
        y_pred = self.fc_out(final_feature)  # (batch, 1)
        return y_pred


class DualStageAttentionRNN(BaseForecaster):
    """Dual-Stage Attention-Based RNN (DA-RNN) for Time Series Forecasting.

    This forecaster implements the DA-RNN model described in [1].
    Used for predicting univariate time series with exogenous features one step ahead.

    Parameters
    ----------
    window_length : int, default=10
        Number of time steps (window size) used for forecasting.
    encoder_hidden_size : int, default=64
        Hidden state size for the encoder LSTM.
    decoder_hidden_size : int, default=64
        Hidden state size for the decoder LSTM.
    attention_dim : int, default=32
        Dimension of the attention space.
    batch_size : int, default=128
        Mini-batch size for training.
    epochs : int, default=100
        Number of training epochs.
    lr : float, default=0.001
        Learning rate.
    device : str, default="cpu"
        Device for PyTorch ("cpu" or "cuda").
    random_state : int or None, default=None
        Random seed.

    References
    ----------
    [1] A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
        Yao Qin et al.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:pred_int": False,
        "capability:multivariate": False,
        "capability:pred_int:insample": False,
        "authors": ["sanskarmodi8"],
    }

    def __init__(
        self,
        window_length=10,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        attention_dim=32,
        batch_size=128,
        epochs=100,
        lr=0.001,
        device="cpu",
        random_state=None,
    ):
        self.window_length = window_length
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_dim = attention_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self._device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.random_state = random_state

        super().__init__()

    def _build_model(self, input_size):
        """Build and initialize the DA-RNN model."""
        self.model_ = DARNNModule(
            input_size,
            self.encoder_hidden_size,
            self.decoder_hidden_size,
            self.attention_dim,
        )
        self.model_.to(self._device)
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr)
        self.scheduler_ = torch.optim.lr_scheduler.StepLR(
            self.optimizer_, step_size=10000, gamma=0.9
        )
        self.criterion_ = nn.MSELoss()

    def _fit(self, y, X, fh=None):
        """
        Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Univariate time series to fit.
        X : pd.DataFrame
            Exogenous time series.
        fh : ForecastingHorizon, optional (default=None)
            Not required by this forecaster.

        Returns
        -------
        self : reference to self
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Store the target variable name for later use
        self._y_name = y.name if hasattr(y, "name") else None

        # Get number of target variables
        self._n_y_columns = 1

        # Convert to numpy arrays
        y = np.asarray(y).reshape(-1, 1)

        # Handle empty or scalar X
        if X is None or (hasattr(X, "shape") and X.shape == ()):
            import warnings

            warnings.warn(
                "X is either empty or scalar. This model requires exogenous variables. "
                "Using dummy values for X, but results may be inaccurate.",
                UserWarning,
            )
            # If X is empty or scalar, create a dummy array with same length as y
            # and a single feature of zeros
            X = np.zeros((len(y), 1))
            X = np.asarray(X)
        else:
            # Ensure X is 2D
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

        # Check if there's enough data
        if len(y) <= self.window_length:
            raise ValueError(
                f"Training data length must be greater \
                    than window_length={self.window_length}"
            )

        n_samples = y.shape[0] - self.window_length
        X_windows = []
        y_history_windows = []
        targets = []
        for i in range(n_samples):
            X_window = X[i : i + self.window_length, :]
            y_hist = y[i : i + self.window_length - 1, :]
            target = y[i + self.window_length - 1, :]
            X_windows.append(X_window)
            y_history_windows.append(y_hist)
            targets.append(target)
        X_windows = np.stack(X_windows)  # (n_samples, window_length, n_features)
        y_history_windows = np.stack(
            y_history_windows
        )  # (n_samples, window_length-1, 1)
        targets = np.stack(targets)  # (n_samples, 1)

        n_features = X_windows.shape[2]
        self._build_model(n_features)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(y_history_windows, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_hist_batch, target_batch in dataloader:
                X_batch = X_batch.to(self._device)
                y_hist_batch = y_hist_batch.to(self._device)
                target_batch = target_batch.to(self._device)

                self.optimizer_.zero_grad()
                y_pred = self.model_(X_batch, y_hist_batch)
                loss = self.criterion_(y_pred, target_batch)
                loss.backward()
                self.optimizer_.step()
                self.scheduler_.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)

        # Store the most recent data for prediction
        self._X_recent = X[-self.window_length :, :]
        self._y_recent = y[-self.window_length + 1 :, :]

        return self

    def _predict(self, X=None, fh=None):
        """
        Forecast time series one step ahead.

        The DA-RNN model uses past target values (y) and both past and current exogenous
        variables X to predict the current target value. For forecasts beyond one step,
        this implementation returns the same one-step forecast for all horizons.

        Parameters
        ----------
        X : pd.DataFrame, optional (default=None)
            Exogenous time series for the forecast horizon.
            If None, a warning is raised as the model is designed to use X.
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to predict.

        Returns
        -------
        y_pred : np.ndarray
            Point forecast values as a 1D array.
        """
        # Handle forecast horizon
        if fh is None:
            n_pred_steps = 1
        else:
            n_pred_steps = len(fh)

        # Create container for predictions
        predictions = np.zeros(n_pred_steps)

        # Check model state variables
        if not hasattr(self, "_X_recent") or not hasattr(self, "_y_recent"):
            raise ValueError(
                "Model state variables (_X_recent, _y_recent) not found. "
                "Ensure the model has been properly fitted."
            )

        # Get the correct X data for prediction
        if X is None:
            import warnings

            warnings.warn(
                "Exogenous series X not provided for prediction. \
                    DA-RNN is designed to use "
                "exogenous variables. Using stored window X_recent\
                     for prediction, but results may be inaccurate.",
                UserWarning,
            )
            # Use stored X_recent for prediction
            X_window = self._X_recent
        else:
            # Process provided X data
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Check if X has the correct number of features
            if X.shape[1] != self._X_recent.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features, but the model was trained with "
                    f"{self._X_recent.shape[1]} features."
                )

            # Use provided X data for prediction
            # For the first prediction, we use the last window_length-1 from X_recent
            # plus the first row from the new X
            X_window = np.vstack((self._X_recent[1:], X[0:1]))

        # Get stored recent y history
        y_history = self._y_recent

        # Convert to PyTorch tensors
        X_tensor = (
            torch.tensor(X_window, dtype=torch.float32).unsqueeze(0).to(self._device)
        )
        y_history_tensor = (
            torch.tensor(y_history, dtype=torch.float32).unsqueeze(0).to(self._device)
        )

        # Make prediction
        self.model_.eval()
        with torch.no_grad():
            one_step_pred = self.model_(X_tensor, y_history_tensor).cpu().numpy().item()

        # If more than one step is requested, warn and fill with the same prediction
        if n_pred_steps > 1:
            import warnings

            warnings.warn(
                "DA-RNN currently only supports one-step-ahead forecasting. "
                "Returning the same forecast for all requested horizons.",
                UserWarning,
            )
            predictions.fill(one_step_pred)
        else:
            predictions[0] = one_step_pred

        return predictions

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the forecaster.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {
                "window_length": 10,
                "encoder_hidden_size": 16,
                "decoder_hidden_size": 16,
                "attention_dim": 8,
                "batch_size": 16,
                "epochs": 5,
                "lr": 0.01,
                "device": "cpu",
            },
            {
                "window_length": 20,
                "encoder_hidden_size": 32,
                "decoder_hidden_size": 32,
                "attention_dim": 16,
                "batch_size": 32,
                "epochs": 10,
                "lr": 0.005,
                "device": "cpu",
            },
        ]
        return params
