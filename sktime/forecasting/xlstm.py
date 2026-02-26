# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""xLSTM forecaster."""

__all__ = ["XLSTMForecaster"]
__author__ = ["muslehal", "vedantag17"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_y


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
        "python_dependencies": "torch>=2.0.0",
        "capability:pred_int": False,
        "capability:pred_var": False,
        "requires-fh-in-fit": False,
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:missing_values": False,
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
        self.block_types = block_types
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sequence_length = sequence_length
        self.device = device

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the xLSTM forecaster to training data."""
        _check_soft_dependencies("torch", severity="error")
        import torch

        from sktime.libs.xlstm_time.xlstm_time import xLSTM

        y = check_y(y)

        # Keep track if input was a Series for prediction output
        self._is_series = isinstance(y, pd.Series)
        if self._is_series:
            self._series_name = y.name
            y = y.to_frame()

        self._y_columns = y.columns
        self._n_outputs = len(self._y_columns)

        y_np = y.values.astype(np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        # Normalize the data
        self.y_mean = np.mean(y_np, axis=0)
        self.y_std = np.std(y_np, axis=0)
        y_normalized = (y_np - self.y_mean) / (self.y_std + 1e-8)

        # Create sequences for training
        sequences, targets = self._create_sequences(y_normalized, self.sequence_length)

        if len(sequences) == 0:
            raise ValueError(
                f"Not enough data. Need at least "
                f"{self.sequence_length + 1} observations."
            )

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve block_types if None
        _block_types = self.block_types
        if _block_types is None:
            _block_types = ["slstm"] * self.num_layers

        # Use actual input size from data
        actual_input_size = y_normalized.shape[1]

        self.model = xLSTM(
            input_size=actual_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            block_types=_block_types,
            num_heads=self.num_heads,
            dropout=self.dropout,
            output_size=actual_input_size,
        ).to(self.device_)

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
                seq_batch = sequences[i : i + self.batch_size].to(self.device_)
                target_batch = targets[i : i + self.batch_size].to(self.device_)

                self.optimizer.zero_grad()
                output = self.model(seq_batch)

                # Output from model is (batch_size, output_size)
                # target_batch is (batch_size, output_size)
                if output.shape != target_batch.shape:
                    if output.ndim > 2:
                        output = output.squeeze(1)
                    if output.shape != target_batch.shape:
                        # Attempt to reshape output to match target
                        output = output.view(target_batch.shape)

                loss = self.loss_fn(output, target_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Store the last sequence for prediction
        self._last_sequence = y_normalized[-self.sequence_length :].copy()

        return self

    def _predict(self, fh, X=None):
        """Generate forecasts for the given forecasting horizon."""
        _check_soft_dependencies("torch", severity="error")
        import torch

        self.model.eval()

        # Get absolute forecasting horizon
        absolute_fh = fh.to_absolute(self.cutoff)

        y_pred = []
        current_sequence = self._last_sequence.copy()

        # Generate predictions step by step
        for _ in range(len(fh)):
            if current_sequence.ndim == 1:
                current_sequence = current_sequence.reshape(-1, 1)

            # Ensure input_seq is 3D: (batch_size=1, sequence_length, features)
            input_seq = torch.tensor(
                current_sequence.reshape(1, self.sequence_length, self._n_outputs),
                dtype=torch.float32,
            ).to(self.device_)

            with torch.no_grad():
                next_val = self.model(input_seq)
                # next_val is roughly (batch_size=1, output_size)
                next_val_np = next_val.cpu().numpy().reshape(self._n_outputs)

            y_pred.append(next_val_np)

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_val_np

        # Denormalize predictions
        y_pred = np.array(y_pred)
        y_pred = y_pred * self.y_std + self.y_mean

        # Return in correct format (Series or DataFrame)
        if hasattr(self, "_is_series") and self._is_series:
            result = pd.Series(
                y_pred.flatten(), index=absolute_fh.to_pandas(), name=self._series_name
            )
        else:
            result = pd.DataFrame(
                y_pred, index=absolute_fh.to_pandas(), columns=self._y_columns
            )

        return result

    def _create_sequences(self, data, seq_length):
        """Create sequences for training."""
        sequences = []
        targets = []

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]

        for i in range(len(data) - seq_length):
            sequences.append(data[i : i + seq_length].reshape(seq_length, n_features))
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
