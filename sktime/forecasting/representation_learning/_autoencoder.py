# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Autoencoder-based representation learning: training wrapper + forecasting glue.

This module is intentionally **not** a network definition. Instead it provides:

- A lightweight wrapper around ``sktime.networks.autoencoder.MLPAutoencoderTorch``
  that implements training loops used by representation-learning forecasters.
- Forecasting-specific utilities such as extracting linear decoder parameters
  for the downstream state-space layer.

In short:
- ``sktime.networks.autoencoder`` = *what the torch model is*
- ``sktime.forecasting.representation_learning._autoencoder`` = *how we train and use it in forecasters*
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sktime.networks.autoencoder import MLPAutoencoderTorch
from sktime.utils.dependencies import _check_soft_dependencies


def _extract_decoder_params(decoder):
    """Extract observation matrix and bias from a linear or extract_params-capable decoder."""
    _check_soft_dependencies("torch", severity="error")
    import torch.nn as nn

    if hasattr(decoder, "extract_params"):
        return decoder.extract_params()
    if isinstance(decoder, nn.Linear):
        w = decoder.weight.data.detach().cpu().numpy()
        b = (
            decoder.bias.data.detach().cpu().numpy()
            if decoder.bias is not None
            else np.zeros(w.shape[0], dtype=w.dtype)
        )
        w = np.nan_to_num(w, nan=0.0)
        return w, b
    raise TypeError(
        "decoder must be nn.Linear or implement extract_params, "
        f"got {type(decoder)}"
    )


class _SimpleAutoencoder:
    """Autoencoder *trainer/wrapper* for forecasters; delegates the network to MLPAutoencoderTorch."""

    def __init__(self, model: MLPAutoencoderTorch):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        encoder_size: tuple[int, ...],
        decoder_size: Optional[tuple[int, ...]] = None,
        decoder_type: str = "linear",
        activation: str = "relu",
        seed: Optional[int] = None,
    ):
        if len(encoder_size) == 0:
            raise ValueError("encoder_size must have at least one element")

        latent_dim = int(encoder_size[-1])
        encoder_hidden = tuple(int(x) for x in encoder_size[:-1])
        decoder_hidden = tuple(int(x) for x in (decoder_size or ()))

        model = MLPAutoencoderTorch(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden,
            decoder_hidden_dims=decoder_hidden,
            decoder_type=str(decoder_type),
            activation=str(activation),
            random_state=int(seed) if seed is not None else None,
        )
        return cls(model=model)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        encoder_size: tuple[int, ...],
        decoder_size: Optional[tuple[int, ...]] = None,
        decoder_type: str = "linear",
        activation: str = "relu",
        seed: Optional[int] = None,
    ):
        return cls.build(
            input_dim=int(dataset.full_input.shape[1]),
            output_dim=int(dataset.y_clean.shape[1]),
            encoder_size=encoder_size,
            decoder_size=decoder_size,
            decoder_type=decoder_type,
            activation=activation,
            seed=seed,
        )

    def __call__(self, x):
        return self.model(x)

    def forward(self, x):
        return self.__call__(x)

    def predict(self, x):
        _check_soft_dependencies("torch", severity="error")
        import torch

        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return list(self.model.parameters())

    def pretrain(
        self,
        full_input,
        y,
        epochs: int,
        batch_size: int,
        optimizer,
        use_mse_loss: bool = True,
    ):
        _check_soft_dependencies("torch", severity="error")
        import torch

        self.train()
        T = int(full_input.shape[0])
        final_epoch_losses = []

        for _epoch in range(int(epochs)):
            epoch_losses = []
            for i in range(0, T, int(batch_size)):
                batch_full_input = full_input[i : i + int(batch_size)]
                batch_y = y[i : i + int(batch_size)]

                optimizer.zero_grad()
                y_pred = self.forward(batch_full_input)

                if use_mse_loss:
                    loss = torch.nn.functional.mse_loss(
                        y_pred, batch_y, reduction="mean"
                    )
                else:
                    mask = ~torch.isnan(batch_y)
                    y_actual = torch.where(
                        torch.isnan(batch_y), torch.zeros_like(batch_y), batch_y
                    )
                    y_pred_masked = y_pred * mask.float()
                    loss = torch.nn.functional.mse_loss(
                        y_pred_masked, y_actual, reduction="mean"
                    )

                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            if epoch_losses:
                final_epoch_losses.append(float(np.mean(epoch_losses)))

        return final_epoch_losses

    def fit(
        self,
        dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer_type: str = "Adam",
        decay_learning_rate: bool = True,
        optimizer=None,
        scheduler=None,
    ):
        _check_soft_dependencies("torch", severity="error")
        import torch

        if optimizer is None:
            if optimizer_type == "SGD":
                optimizer = torch.optim.SGD(self.parameters(), lr=float(learning_rate))
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))

        if scheduler is None and decay_learning_rate:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.99
            )

        self.train()
        full_input = dataset.full_input
        y_clean = dataset.y_clean
        T = int(len(dataset))

        for _epoch in range(int(epochs)):
            for i in range(0, T, int(batch_size)):
                batch_input = full_input[i : i + int(batch_size)]
                batch_target = y_clean[i : i + int(batch_size)]

                optimizer.zero_grad()
                pred = self.forward(batch_input)

                mask = ~torch.isnan(batch_target)
                y_actual_ = torch.where(
                    torch.isnan(batch_target),
                    torch.zeros_like(batch_target),
                    batch_target,
                )
                y_predicted_ = pred * mask.float()
                loss = torch.nn.functional.mse_loss(
                    y_predicted_, y_actual_, reduction="mean"
                )
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
