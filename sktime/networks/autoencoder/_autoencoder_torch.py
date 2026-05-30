"""Autoencoder *network* structures in PyTorch.

This module is intentionally **architecture-only**: it defines reusable torch
``nn.Module`` building blocks (e.g., an MLP autoencoder) that can be shared
across estimators.

Training loops, dataset adapters, and forecasting-specific helpers live outside
``sktime.networks`` (e.g., in ``sktime.forecasting.representation_learning``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


def _instantiate_activation(act: str | Callable | None):
    if act is None:
        return None
    if not isinstance(act, str):
        # assume callable/torch module
        return act
    name = act.lower()
    if name == "relu":
        return _safe_import("torch.nn.ReLU")()
    if name == "tanh":
        return _safe_import("torch.nn.Tanh")()
    if name == "gelu":
        return _safe_import("torch.nn.GELU")()
    if name == "sigmoid":
        return _safe_import("torch.nn.Sigmoid")()
    raise ValueError(f"Unknown activation: {act!r}")


class MLPAutoencoderTorch(NNModule):
    """Simple MLP autoencoder in PyTorch.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int, optional (default=None)
        Output feature dimension. If None, defaults to ``input_dim``.
    latent_dim : int
        Latent feature dimension.
    encoder_hidden_dims : tuple[int, ...], optional (default=())
        Hidden layer sizes for the encoder (excluding the latent layer).
    decoder_hidden_dims : tuple[int, ...], optional (default=())
        Hidden layer sizes for the decoder (excluding the output layer).
        Only used when ``decoder_type="mlp"``.
    decoder_type : {"linear", "mlp"}, optional (default="linear")
        Decoder type.
    activation : str or callable or None, optional (default="relu")
        Activation function between layers.
    bias : bool, optional (default=True)
        Whether Linear layers use bias.
    random_state : int or None, optional (default=None)
        Seed for torch initialization.
    """

    _tags = {
        "authors": ["minkeymouse"],
        "maintainers": ["minkeymouse"],
        "python_dependencies": ["torch"],
        "capability:random_state": True,
        "property:randomness": "stochastic",
    }

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int | None = None,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...] = (),
        decoder_hidden_dims: tuple[int, ...] = (),
        decoder_type: str = "linear",
        activation: str | Callable | None = "relu",
        bias: bool = True,
        random_state: int | None = None,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = tuple(int(x) for x in encoder_hidden_dims)
        self.decoder_hidden_dims = tuple(int(x) for x in decoder_hidden_dims)
        self.decoder_type = str(decoder_type)
        self.activation = activation
        self.bias = bool(bias)
        self.random_state = int(random_state) if random_state is not None else None
        super().__init__()

        if self.input_dim < 1 or self.output_dim < 1 or self.latent_dim < 1:
            raise ValueError("input_dim, output_dim and latent_dim must be >= 1")
        if self.decoder_type not in ("linear", "mlp"):
            raise ValueError("decoder_type must be 'linear' or 'mlp'")

        torch = _safe_import("torch")
        nn = _safe_import("torch.nn")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        act = _instantiate_activation(self.activation)

        # Encoder: [input] -> hidden* -> latent
        enc_layers: list[nn.Module] = []
        prev = self.input_dim
        for h in self.encoder_hidden_dims:
            enc_layers.append(nn.Linear(prev, h, bias=self.bias))
            if act is not None:
                enc_layers.append(act)
            prev = h
        enc_layers.append(nn.Linear(prev, self.latent_dim, bias=self.bias))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent -> hidden* -> output_dim
        if self.decoder_type == "linear":
            self.decoder = nn.Linear(self.latent_dim, self.output_dim, bias=self.bias)
            self._decoder_last_linear = self.decoder
        else:
            dec_layers: list[nn.Module] = []
            prev = self.latent_dim
            for h in self.decoder_hidden_dims:
                dec_layers.append(nn.Linear(prev, h, bias=self.bias))
                if act is not None:
                    dec_layers.append(act)
                prev = h
            out = nn.Linear(prev, self.output_dim, bias=self.bias)
            dec_layers.append(out)
            self.decoder = nn.Sequential(*dec_layers)
            self._decoder_last_linear = out

        # initialize weights similar to other network modules
        init_xavier = _safe_import("torch.nn.init.xavier_normal_")
        init_const = _safe_import("torch.nn.init.constant_")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_xavier(m.weight, gain=1.0)
                if m.bias is not None:
                    init_const(m.bias, 0.0)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = _safe_import("torch.from_numpy")(X).float()
        return self.decode(self.encode(X))

    def get_last_linear_layer(self):
        """Return last linear layer of decoder (for parameter extraction)."""
        return self._decoder_last_linear

    def extract_params(self):
        """Return decoder output-layer weight and bias as numpy arrays."""
        layer = self.get_last_linear_layer()
        w = layer.weight.data.detach().cpu().numpy()
        b = (
            layer.bias.data.detach().cpu().numpy()
            if layer.bias is not None
            else np.zeros(w.shape[0], dtype=w.dtype)
        )
        w = np.nan_to_num(w, nan=0.0)
        return w, b

