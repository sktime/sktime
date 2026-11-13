"""Residual Network (ResNet) for Classification and Regression in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["ResNetNetworkTorch"]

from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class ResNetNetworkTorch(NNModule):
    """Establish the network structure for a ResNet in PyTorch.

    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    Parameters
    ----------
    input_size : int or tuple of int
        Number of expected features in the input. If tuple, must be of length 3
        and in format (n_instances, n_dims, series_length).
    num_classes : int
        Number of outputs.
    n_filters : tuple of int, default = (64, 128, 128)
        Number of convolutional filters in each residual block. The length of
        this tuple determines the number of residual blocks. Any length >= 1
        is allowed.
    kernel_size : tuple of int, default = (8, 5, 3)
        Length of the 1D convolution window for the conv layers within a
        residual block, shared across all residual blocks. Any length >= 1
        is allowed.
    activation : callable or None, default = None
        Activation function to use in the output layer.
    activation_hidden : callable or None, default = None
        Activation function to use in the hidden layers, and after each
        residual connection.
    random_state : int or None, default=None
        Seed to ensure reproducibility.
    init_weights : bool, default = True
        Whether to apply custom initialization.

    References
    ----------
    .. [1] Wang et al, Time series classification from scratch with deep
    neural networks: A strong baseline, International joint conference on
    neural networks (IJCNN), 2017.
    """

    _tags = {
        "authors": ["srupat"],
        "maintainers": ["srupat"],
        "python_version": ">=3.10, <3.15",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int | tuple[int, ...],
        num_classes: int,
        n_filters: tuple[int, ...] = (64, 128, 128),
        kernel_size: tuple[int, ...] = (8, 5, 3),
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
        random_state: int | None = None,
        init_weights: bool = True,
    ):
        super().__init__()

        self._import_cache = {}
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.random_state = random_state
        self.init_weights = init_weights

        if not isinstance(self.n_filters, tuple) or not all(
            isinstance(f, int) for f in self.n_filters
        ):
            raise TypeError("`n_filters` must be a tuple of ints.")
        if len(self.n_filters) < 1:
            raise ValueError("`n_filters` must have length >= 1.")

        if not isinstance(self.kernel_size, tuple) or not all(
            isinstance(k, int) for k in self.kernel_size
        ):
            raise TypeError("`kernel_size` must be a tuple of ints.")
        if len(self.kernel_size) < 1:
            raise ValueError("`kernel_size` must have length >= 1.")

        if isinstance(self.input_size, int):
            n_dims = self.input_size
        elif isinstance(self.input_size, tuple):
            if len(self.input_size) == 3:
                _, n_dims, _ = self.input_size
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must either be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(self.input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(self.input_size)}"
            )
        self.n_dims = n_dims

        ModuleList = _safe_import("torch.nn.ModuleList")
        self.conv_blocks = ModuleList()
        self.shortcuts = ModuleList()
        in_channels = self.n_dims
        for out_channels in self.n_filters:
            self.conv_blocks.append(self._make_conv_block(in_channels, out_channels))
            self.shortcuts.append(self._make_shortcut(in_channels, out_channels))
            in_channels = out_channels

        AdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        self.gap = AdaptiveAvgPool1d(1)

        Linear = _safe_import("torch.nn.Linear")
        self.out = Linear(in_channels, self.num_classes)

        if self.init_weights:
            self.apply(self._init_weights)

    def _torch_op(self, import_path):
        """Lazy import and cache torch ops used in forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _make_conv_block(self, in_channels, out_channels):
        """Build the stack of conv layers making up a residual block."""
        Sequential = _safe_import("torch.nn.Sequential")
        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        layers = []
        channels = in_channels
        for i, k in enumerate(self.kernel_size):
            layers.append(Conv1d(channels, out_channels, kernel_size=k, padding="same"))
            layers.append(BatchNorm1d(out_channels))
            if i < len(self.kernel_size) - 1 and self.activation_hidden is not None:
                layers.append(self.activation_hidden)
            channels = out_channels
        return Sequential(*layers)

    def _make_shortcut(self, in_channels, out_channels):
        """Build the shortcut connection for a residual block."""
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        if in_channels == out_channels:
            return BatchNorm1d(out_channels)
        Sequential = _safe_import("torch.nn.Sequential")
        Conv1d = _safe_import("torch.nn.Conv1d")
        return Sequential(
            Conv1d(in_channels, out_channels, kernel_size=1, padding="same"),
            BatchNorm1d(out_channels),
        )

    def _init_weights(self, module):
        """Apply tensorflow-like initializations.

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply the initialization.
        """
        Conv1d = _safe_import("torch.nn.Conv1d")
        Linear = _safe_import("torch.nn.Linear")
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        zeros_ = _safe_import("torch.nn.init.zeros_")
        if isinstance(module, (Conv1d, Linear)):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input tensor containing the time series data.
        """
        if isinstance(X, np.ndarray):
            torch_from_numpy = self._torch_op("torch.from_numpy")
            X = torch_from_numpy(X).float()

        x = X.transpose(1, 2)  # (batch_size, n_dims, seq_length) for Conv1d
        for conv_block, shortcut in zip(self.conv_blocks, self.shortcuts):
            residual = shortcut(x)
            x = conv_block(x) + residual
            if self.activation_hidden is not None:
                x = self.activation_hidden(x)

        x = self.gap(x).squeeze(-1)
        x = self.out(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
