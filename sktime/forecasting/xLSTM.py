# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements xLSMT forecaster.

uses the existing code from https://github.com/muslehal/xLSTMTime.
"""

__all__ = ["xLSTM"]
__author__ = ["muslehal", "vedantag17"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import
from sktime.utils.validation.forecasting import check_y

nn = _safe_import("torch.nn")
torch = _safe_import("torch")


class sLSTMBlock(nn.Module):
    """Stabilized LSTM Block."""

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Input, forget, and output gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_z = nn.Linear(input_size, hidden_size)

        # Recurrent connections
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Layer normalization
        self.ln_i = nn.LayerNorm(hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the sLSTM block."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            n = torch.zeros(batch_size, self.hidden_size, device=x.device)
            m = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c, n, m = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Compute gates
            i_t = torch.sigmoid(self.ln_i(self.W_i(x_t) + self.R_i(h)))
            f_t = torch.sigmoid(self.ln_f(self.W_f(x_t) + self.R_f(h)))
            o_t = torch.sigmoid(self.ln_o(self.W_o(x_t) + self.R_o(h)))
            z_t = torch.tanh(self.ln_z(self.W_z(x_t) + self.R_z(h)))

            # Update cell state with stabilization
            m = torch.max(f_t + i_t, m)
            i_t_hat = torch.exp(i_t - m)
            f_t_hat = torch.exp(f_t - m)

            c = f_t_hat * c + i_t_hat * z_t
            n = f_t_hat * n + i_t_hat

            # Stabilized cell state
            c_tilde = c / n
            h = o_t * torch.tanh(c_tilde)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c, n, m)


class mLSTMBlock(nn.Module):
    """Matrix LSTM Block."""

    def __init__(self, input_size, hidden_size, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        # Input and forget gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)

        # Layer normalization
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_k = nn.LayerNorm(hidden_size)
        self.ln_v = nn.LayerNorm(hidden_size)

        # Output projection
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the mLSTM block."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            C = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
            )
            n = torch.zeros(batch_size, self.num_heads, self.head_dim, device=x.device)
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            C, n, h = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Compute projections
            q_t = self.ln_q(self.W_q(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            k_t = self.ln_k(self.W_k(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            v_t = self.ln_v(self.W_v(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )

            # Compute gates
            i_t = torch.sigmoid(self.W_i(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            f_t = torch.sigmoid(self.W_f(x_t)).view(
                batch_size, self.num_heads, self.head_dim
            )
            o_t = torch.sigmoid(self.W_o(x_t))

            # Update matrix memory
            C = f_t.unsqueeze(-1) * C + i_t.unsqueeze(-1) * torch.bmm(
                v_t.unsqueeze(-1), k_t.unsqueeze(-2)
            )
            n = f_t * n + i_t * k_t

            # Compute output
            h_heads = torch.bmm(C, q_t.unsqueeze(-1)).squeeze(-1) / (
                n.unsqueeze(-1) + 1e-8
            )
            h_concat = h_heads.view(batch_size, -1)
            h = o_t * self.proj(h_concat)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (C, n, h)


class xLSTMBlock(nn.Module):
    """Extended LSTM Block (can be either sLSTM or mLSTM)."""

    def __init__(
        self, input_size, hidden_size, block_type="slstm", num_heads=1, dropout=0.0
    ):
        super().__init__()
        self.block_type = block_type

        if block_type == "slstm":
            self.block = sLSTMBlock(input_size, hidden_size, num_heads)
        elif block_type == "mlstm":
            self.block = mLSTMBlock(input_size, hidden_size, num_heads)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, state=None):
        """Forward pass through the xLSTM block."""
        # Residual connection
        residual = x if x.shape[-1] == self.block.hidden_size else None

        output, new_state = self.block(x, state)
        output = self.dropout(output)

        # Add residual connection if dimensions match
        if residual is not None:
            output = output + residual

        output = self.norm(output)
        return output, new_state


class xLSTM(nn.Module):
    """Extended Long Short-Term Memory Network."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        block_types=None,
        num_heads=1,
        dropout=0.0,
        output_size=1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        if block_types is None:
            block_types = ["slstm"] * num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # xLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size
            self.layers.append(
                xLSTMBlock(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    block_type=block_types[i % len(block_types)],
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, states=None):
        """Forward pass through the xLSTM network."""
        # Input projection
        x = self.input_proj(x)

        if states is None:
            states = [None] * self.num_layers

        new_states = []

        # Pass through xLSTM layers
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i])
            new_states.append(new_state)

        # Output projection (use last timestep)
        output = self.output_proj(x[:, -1, :])

        return output


class XLSTMForecaster(BaseForecaster):
    """xLSTM Forecaster for time series prediction using Extended LSTM architecture.

    Parameters
    ----------
    input_size : int, default=1
        Number of input features
    hidden_size : int, default=64
        Hidden state size for xLSTM blocks
    num_layers : int, default=2
        Number of xLSTM layers
    block_types : list, default=None
        List of block types ('slstm' or 'mlstm'). If None, uses all 'slstm'
    num_heads : int, default=1
        Number of attention heads for mLSTM blocks
    dropout : float, default=0.1
        Dropout probability
    learning_rate : float, default=0.001
        Learning rate for optimization
    batch_size : int, default=32
        Batch size for training
    n_epochs : int, default=50
        Number of training epochs
    sequence_length : int, default=20
        Length of input sequences
    device : str, default=None
        Device to use ('cuda' or 'cpu'). If None, auto-detects

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.xlstm import XLSTMForecaster
    >>> y = load_airline()
    >>> forecaster = XLSTMForecaster(
    ...     hidden_size=32,
    ...     num_layers=2,
    ...     n_epochs=10
    ... )
    >>> forecaster.fit(y)
    XLSTMForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "python_dependencies": "torch>=2.0.0,<2.4.0",
        "capability:pred_int": False,
        "capability:pred_var": False,
        "requires-fh-in-fit": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "handles-missing-data": False,
        "requires-positive-X": False,
    }

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        block_types=None,
        num_heads=1,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32,
        n_epochs=50,
        sequence_length=20,
        device=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_types = block_types or ["slstm"] * num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sequence_length = sequence_length

        super().__init__()
        import torch

        self.device = self._device_param or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _fit(self, y, X=None, fh=None):
        """Fit the xLSTM forecaster to training data."""
        import torch

        y = check_y(y)
        y_np = y.values.astype(np.float32)

        # Normalize the data
        self.y_mean = np.mean(y_np)
        self.y_std = np.std(y_np)
        y_normalized = (y_np - self.y_mean) / (self.y_std + 1e-8)

        # Create sequences for training
        sequences, targets = self._create_sequences(y_normalized, self.sequence_length)

        if len(sequences) == 0:
            raise ValueError(
                f"Not enough data. Need at least "
                f"{self.sequence_length + 1} observations."
            )

        # Initialize model
        self.model = xLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            block_types=self.block_types,
            num_heads=self.num_heads,
            dropout=self.dropout,
            output_size=1,
        ).to(self.device)

        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.loss_fn = torch.nn.MSELoss()

        # Convert to tensors
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(sequences), self.batch_size):
                seq_batch = sequences[i : i + self.batch_size].to(self.device)
                target_batch = targets[i : i + self.batch_size].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(seq_batch)
                loss = self.loss_fn(output.squeeze(), target_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Store the last sequence for prediction
        self._last_sequence = y_normalized[-self.sequence_length :]

        return self

    def _predict(self, fh, X=None):
        """Generate forecasts for the given forecasting horizon."""
        self.model.eval()

        # Get absolute forecasting horizon
        absolute_fh = fh.to_absolute(self.cutoff)

        y_pred = []
        current_sequence = self._last_sequence.copy()

        # Generate predictions step by step
        for _ in range(len(fh)):
            # Prepare input
            input_seq = torch.tensor(
                current_sequence.reshape(1, -1, 1), dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                next_val = self.model(input_seq)
                next_val_np = next_val.cpu().numpy().item()

            y_pred.append(next_val_np)

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_val_np

        # Denormalize predictions
        y_pred = np.array(y_pred) * self.y_std + self.y_mean

        # Return as pandas Series with correct index
        return pd.Series(y_pred, index=absolute_fh.to_pandas(), name="predictions")

    def _create_sequences(self, data, seq_length):
        """Create sequences for training."""
        sequences = []
        targets = []

        for i in range(len(data) - seq_length):
            sequences.append(data[i : i + seq_length].reshape(-1, 1))
            targets.append(data[i + seq_length])

        return np.array(sequences), np.array(targets)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # Minimal parameters for fast testing
        params1 = {
            "hidden_size": 8,
            "num_layers": 1,
            "block_types": ["slstm"],
            "num_heads": 1,
            "dropout": 0.0,
            "learning_rate": 0.01,
            "batch_size": 4,
            "n_epochs": 2,
            "sequence_length": 3,
        }

        # Alternative configuration with mLSTM
        params2 = {
            "hidden_size": 4,
            "num_layers": 2,
            "block_types": ["slstm", "mlstm"],
            "num_heads": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 2,
            "n_epochs": 1,
            "sequence_length": 2,
        }

        # Minimal single layer for edge case testing
        params3 = {
            "hidden_size": 4,
            "num_layers": 1,
            "block_types": ["mlstm"],
            "num_heads": 1,
            "dropout": 0.0,
            "learning_rate": 0.1,
            "batch_size": 1,
            "n_epochs": 1,
            "sequence_length": 2,
        }

        return [params1, params2, params3]
