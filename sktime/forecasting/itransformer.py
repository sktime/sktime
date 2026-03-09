# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""iTransformer Forecaster."""

__author__ = ["TenFinges"]

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset
else:
    # dummy class to avoid import errors
    class Dataset:
        """Dummy Dataset class to avoid import errors."""

        pass


class ITransformerDataset(Dataset):
    """Dataset for iTransformer."""

    def __init__(self, y, context_length, prediction_length):
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Handle multi-index or single series
        if isinstance(y.index, pd.MultiIndex):
            # Group by instance
            self.series = []
            for _, group in y.groupby(level=0):
                self.series.append(group.values)
        else:
            self.series = [y.values]

        self.samples = []
        for i, serie in enumerate(self.series):
            # Create sliding windows
            # serie shape: (Time, Variates)
            length = serie.shape[0]
            if length <= context_length + prediction_length:
                # Pad or skip? For now, skip if too short, or maybe pad?
                # Using simple sliding window for now.
                continue

            for t in range(length - context_length - prediction_length + 1):
                self.samples.append((i, t))

    def __len__(self):
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return item."""
        series_idx, start_idx = self.samples[idx]
        serie = self.series[series_idx]

        # Context window
        x = serie[start_idx : start_idx + self.context_length]
        # Prediction window
        y = serie[
            start_idx + self.context_length : start_idx
            + self.context_length
            + self.prediction_length
        ]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class ITransformerForecaster(BaseDeepNetworkPyTorch):
    """iTransformer Forecaster.

    Implementation of the iTransformer model from "iTransformer: Inverted Transformers
    are Effective for Time Series Forecasting".

    Parameters
    ----------
    context_length : int, default=96
        Length of the input sequence (lookback window).
    prediction_length : int, default=96
        Length of the prediction sequence (forecast horizon).
    num_epochs : int, default=10
        Number of epochs to train.
    batch_size : int, default=32
        Number of training examples per batch.
    d_model : int, default=512
        Dimension of the transformer hidden state.
    nhead : int, default=8
        Number of attention heads.
    num_encoder_layers : int, default=2
        Number of transformer encoder layers.
    dim_feedforward : int, default=2048
        Dimension of the feedforward network model.
    dropout : float, default=0.1
        Dropout value.
    activation : str, default='relu'
        Activation function of the transformer encoder.
    optimizer : torch.optim.Optimizer, default=torch.optim.Adam
        Optimizer to be used for training.
    lr : float, default=0.001
        Learning rate.
    """

    _tags = {
        "authors": ["TenFinges"],
        "maintainers": ["TenFinges"],
        "python_dependencies": ["torch"],
        "capability:global_forecasting": True,
        "tests:vm": True,
    }

    def __init__(
        self,
        context_length=96,
        prediction_length=96,
        num_epochs=10,
        batch_size=32,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        in_channels=1,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.in_channels = in_channels
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            criterion_kwargs=criterion_kwargs,
            lr=lr,
            in_channels=in_channels,
        )

        if _check_soft_dependencies("torch"):
            import torch

            self.optimizers = {
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "SGD": torch.optim.SGD,
            }

            self.criterions = {
                "MSE": torch.nn.MSELoss,
                "L1": torch.nn.L1Loss,
                "SmoothL1": torch.nn.SmoothL1Loss,
                "Huber": torch.nn.HuberLoss,
            }  # Standard MSE loss

    def _build_network(self, fh):
        """Build the iTransformer Network."""
        from sktime.networks.itransformer import ITransformerNetwork

        return ITransformerNetwork(
            seq_len=self.context_length,
            pred_len=self.prediction_length,
            num_variates=self.in_channels,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )._build()

    def build_pytorch_train_dataloader(self, y):
        """Build PyTorch DataLoader for training."""
        dataset = ITransformerDataset(y, self.context_length, self.prediction_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def build_pytorch_pred_dataloader(self, y, fh):
        """Build PyTorch DataLoader for prediction."""
        # For prediction, we need the last `context_length` window
        # We assume y contains enough history.
        # This part requires careful handling of indices to match fh.
        # But BaseDeepNetworkPyTorch usually calls predict with X?
        # Actually, base class _predict typically just passes the last sequence.

        # Re-implementing simplified prediction loader logic:
        # We just need the lookback window ending at cutoff.

        # Convert y to numpy buffer
        if isinstance(y.index, pd.MultiIndex):
            series = [group.values for _, group in y.groupby(level=0)]
        else:
            series = [y.values]

        X_pred = []
        for s in series:
            # Take last context_length points
            if len(s) < self.context_length:
                # Pad with zeros? or repeat?
                pad = np.zeros((self.context_length - len(s), s.shape[1]))
                s_padded = np.concatenate([pad, s], axis=0)
                X_pred.append(s_padded)
            else:
                X_pred.append(s[-self.context_length :])

        X_pred = torch.tensor(np.array(X_pred), dtype=torch.float32)

        # Create a dummy dataset/loader
        class PredDataset(Dataset):
            """Dataset for prediction."""

            def __init__(self, X):
                """Initialize the prediction dataset.

                Parameters
                ----------
                X : torch.Tensor
                    The input tensor for prediction.
                """
                self.X = X

            def __len__(self):
                """Return length of dataset."""
                return len(self.X)

            def __getitem__(self, i):
                """Return item.

                Parameters
                ----------
                i : int
                    Index of the sample to retrieve.

                Returns
                -------
                tuple
                    (input tensor, 0.0)
                """
                return self.X[i], 0.0

        return DataLoader(
            PredDataset(X_pred),
            batch_size=self.batch_size,
            shuffle=False,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = [
            {
                "context_length": 4,
                "prediction_length": 6,
                "num_epochs": 1,
                "batch_size": 4,
                "d_model": 16,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 32,
            },
            {
                "context_length": 5,
                "prediction_length": 6,
                "num_epochs": 1,
                "batch_size": 2,
                "d_model": 8,
                "nhead": 1,
                "dim_feedforward": 16,
            },
        ]
        return params
