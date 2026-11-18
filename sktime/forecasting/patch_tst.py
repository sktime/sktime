# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Native PatchTST forecaster implemented with PyTorch."""

__all__ = ["PatchTSTForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


def _torch():
    """Import torch only when the estimator is used."""
    _check_soft_dependencies("torch", severity="error")
    import torch

    return torch


class _PatchTSTNetwork(nn.Module):
    """Shared channel-independent PatchTST backbone and forecasting head."""

    def __init__(self, patch_len, stride, d_model, n_heads, dropout, horizon):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embedding = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, values):
        patches = values.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        batch, channels, n_patches, _ = patches.shape
        tokens = self.patch_embedding(patches.reshape(-1, n_patches, self.patch_len))
        encoded = self.backbone(tokens).mean(dim=1)
        forecast = self.head(encoded).reshape(batch, channels, -1)
        return forecast.transpose(1, 2)


class PatchTSTForecaster(BaseForecaster):
    """PatchTST forecaster with channel-independent Transformer processing.

    Parameters
    ----------
    patch_len : int, default=16
        Length of each temporal patch.
    stride : int, default=8
        Step between consecutive patches.
    d_model : int, default=64
        Transformer embedding width.
    n_heads : int, default=4
        Number of self-attention heads.
    dropout : float, default=0.1
        Transformer dropout probability.
    context_length : int, default=32
        Number of observations used for each training example.
    max_epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=32
        Mini-batch size.
    learning_rate : float, default=1e-3
        Adam learning rate.
    warmup_steps : int, default=0
        Number of linear learning-rate warmup steps.
    random_state : int or None, default=None
        Seed for reproducible training.
    device : str or None, default=None
        Torch device. Defaults to CUDA when available, otherwise CPU.

    Examples
    --------
    >>> forecaster = PatchTSTForecaster(patch_len=4, stride=2)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["sktime developers"],
        "python_dependencies": ["torch"],
        "category": "deep_learning",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "tests:libs": ["torch"],
    }

    def __init__(
        self,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        dropout=0.1,
        context_length=32,
        max_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        warmup_steps=0,
        random_state=None,
        device=None,
    ):
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.context_length = context_length
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.random_state = random_state
        self.device = device
        super().__init__()

    def __post_init__(self):
        """Validate hyper-parameters without importing torch."""
        if self.patch_len < 1 or self.stride < 1:
            raise ValueError("patch_len and stride must be positive")
        if self.patch_len > self.context_length:
            raise ValueError("patch_len must not exceed context_length")
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError(
                "max_epochs, batch_size, and learning_rate must be positive"
            )
        self._device = None

    def _make_model(self, n_channels, prediction_length):
        return _PatchTSTNetwork(
            self.patch_len,
            self.stride,
            self.d_model,
            self.n_heads,
            self.dropout,
            prediction_length,
        )

    def _fit(self, y, X=None, fh=None):
        torch = _torch()
        self._cur_y = y
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        values = torch.as_tensor(y.to_numpy(dtype=np.float32))
        horizon = (
            int(max(fh.to_relative(self.cutoff)._values)) if fh is not None else 1
        )
        if len(values) < self.context_length + horizon:
            raise ValueError(
                "y is too short for context_length and forecasting horizon"
            )
        self._device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_ = self._make_model(values.shape[1], horizon).to(self._device)
        xs, ys = [], []
        for end in range(self.context_length, len(values) - horizon + 1):
            xs.append(values[end - self.context_length : end])
            ys.append(values[end : end + horizon])
        x_train = torch.stack(xs).to(self._device)
        y_train = torch.stack(ys).to(self._device)
        self.model_.patches_seen_ = len(xs)
        if _check_soft_dependencies("lightning", severity="none"):
            import lightning.pytorch as pl
            from torch.utils.data import DataLoader, TensorDataset

            forecaster = self

            class LightningPatchTST(pl.LightningModule):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def training_step(self, batch, batch_idx):
                    predictions = self.network(batch[0])
                    return torch.nn.functional.mse_loss(predictions, batch[1])

                def configure_optimizers(self):
                    optimizer = torch.optim.Adam(
                        self.network.parameters(), lr=forecaster.learning_rate
                    )
                    if not forecaster.warmup_steps:
                        return optimizer
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lambda step: min(1.0, (step + 1) / forecaster.warmup_steps),
                    )
                    return {"optimizer": optimizer, "lr_scheduler": scheduler}

            module = LightningPatchTST(self.model_)
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=self.max_epochs,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(
                module,
                DataLoader(
                    TensorDataset(x_train, y_train),
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
            )
            self.model_ = module.network
            return self

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        if self.warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
        step = 0
        self.model_.train()
        for _ in range(self.max_epochs):
            order = torch.randperm(len(x_train), device=self._device)
            for batch_indices in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    self.model_(x_train[batch_indices]), y_train[batch_indices]
                )
                loss.backward()
                if self.warmup_steps:
                    scale = min(1.0, (step + 1) / self.warmup_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = self.learning_rate * scale
                optimizer.step()
                step += 1
        return self

    def _predict(self, fh, X=None):
        torch = _torch()
        values = torch.as_tensor(self._cur_y.to_numpy(dtype=np.float32))
        context = values[-self.context_length :].unsqueeze(0).to(self._device)
        self.model_.eval()
        with torch.no_grad():
            all_predictions = self.model_(context)[0].cpu().numpy()
        relative = fh.to_relative(self.cutoff)._values.astype(int)
        prediction = all_predictions[relative - 1]
        index = fh.to_absolute(self.cutoff).to_pandas()
        return pd.DataFrame(prediction, index=index, columns=self._cur_y.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return two small configurations for estimator checks."""
        params = {
            "patch_len": 2,
            "stride": 1,
            "d_model": 8,
            "n_heads": 2,
            "dropout": 0.0,
            "context_length": 8,
            "max_epochs": 1,
            "batch_size": 4,
        }
        return [params, {**params, "patch_len": 4, "stride": 2}]
