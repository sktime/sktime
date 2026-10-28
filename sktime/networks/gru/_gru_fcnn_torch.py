"""GRU with fully convolutional network (GRU-FCN) in PyTorch."""

__authors__ = ["NellyElsayed", "fnhirwa"]
__all__ = ["GRUFCNNNetworkTorch"]

from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class GRUFCNNNetworkTorch(NNModule):
    """GRU with fully convolutional network, for time series classification.

    A GRU branch and a fully convolutional branch are applied to the input
    series in parallel. The last hidden state of the GRU and the globally
    average pooled convolutional features are concatenated and passed to a
    single fully connected output layer.

    This network was originally defined in [1]_. The implementation uses
    PyTorch and references the TensorFlow implementations in [2]_ and [3]_.

    Parameters
    ----------
    input_size : int
        Number of features in the input time series.
    num_classes : int
        Number of outputs, i.e., the number of classes. Use ``1`` for regression.
    hidden_dim : int, default=256
        Number of features in the hidden state of the GRU.
    gru_layers : int, default=4
        Number of GRU layers.
    batch_first : bool, default=False
        If True, then the input tensor is provided as (batch, seq, feature),
        otherwise as (seq, batch, feature).
    bias : bool, default=True
        If False, then the GRU does not use bias weights.
    init_weights : bool, default=True
        If True, then the weights are initialized as in the reference
        implementation.
    dropout : float, default=0.0
        Dropout rate applied between the GRU layers.
    gru_dropout : float, default=0.0
        Dropout rate applied to the GRU output.
    bidirectional : bool, default=False
        If True, then the GRU is bidirectional.
    conv_layers : tuple of int, default=(128, 256, 128)
        Number of filters in each convolutional layer. The number of
        convolutional layers equals the length of this tuple.
    kernel_sizes : tuple of int, default=(7, 5, 3)
        Kernel size of each convolutional layer. Must have the same length
        as ``conv_layers``.
    activation : Callable or None, default=None
        Activation applied to the output layer. If None, the network returns
        raw outputs, i.e., logits.
    activation_hidden : Callable or None, default=None
        Activation applied in the convolutional blocks. If None, no activation
        is applied.

    References
    ----------
    .. [1] Elsayed, et al. "Deep Gated Recurrent and Convolutional Network Hybrid Model
        for Univariate Time Series Classification."
        arXiv preprint arXiv:1812.07683 (2018).
    .. [2] https://github.com/NellyElsayed/GRU-FCN-model-for-univariate-time-series-classification
    .. [3] https://github.com/titu1994/LSTM-FCN
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["NellyElsayed", "fnhirwa"],
        "maintainers": ["fnhirwa", "srupat"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "GRUFCNNNetworkTorch",
        input_size: int,
        num_classes: int,
        hidden_dim: int = 256,
        gru_layers: int = 4,
        batch_first: bool = False,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
        conv_layers: tuple[int, ...] = (128, 256, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
    ):
        super().__init__()

        nnGRU = _safe_import("torch.nn.GRU")
        nnSequential = _safe_import("torch.nn.Sequential")
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnIdentity = _safe_import("torch.nn.Identity")
        nnLinear = _safe_import("torch.nn.Linear")

        conv_layers = tuple(conv_layers)
        kernel_sizes = tuple(kernel_sizes)
        if len(conv_layers) != len(kernel_sizes):
            raise ValueError(
                "`conv_layers` and `kernel_sizes` must have the same length, "
                f"but got {len(conv_layers)} and {len(kernel_sizes)}."
            )

        self._import_cache = {}

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.gru_dropout = gru_dropout
        self.bidirectional = bidirectional
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes
        self.init_weights = init_weights

        # GRU branch
        self.gru = nnGRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # fully convolutional branch; each block is conv, batch norm, activation
        act_hidden = nnIdentity() if activation_hidden is None else activation_hidden
        in_channels = (input_size,) + conv_layers[:-1]
        self.conv_blocks = _safe_import("torch.nn.ModuleList")(
            [
                nnSequential(
                    nnConv1d(
                        in_channels=n_in,
                        out_channels=n_out,
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nnBatchNorm1d(n_out),
                    act_hidden,
                )
                for n_in, n_out, kernel_size in zip(
                    in_channels, conv_layers, kernel_sizes
                )
            ]
        )
        self.globalavgpool = nnAdaptiveAvgPool1d(1)

        # combined output layer
        self.grudropout = nnDropout(gru_dropout) if gru_dropout else nnIdentity()
        self.fc = nnLinear(
            hidden_dim * (1 + bidirectional) + conv_layers[-1], num_classes
        )
        self.activation = activation

        if self.init_weights:
            self.apply(self._init_gru_weights)
            self.apply(self._init_conv_weights)

    def _torch_op(self, import_path):
        """Lazily import and cache a torch callable used in the forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _init_gru_weights(self, module):
        """Apply a TensorFlow like Glorot uniform initialization in place.

        Adapted from
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization

        Parameters
        ----------
        module : torch.nn.Module
            Module whose parameters are initialized in place.
        """
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")

        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def _init_conv_weights(self, module):
        """Apply the initialization of the original paper in place.

        Parameters
        ----------
        module : torch.nn.Module
            Module whose parameters are initialized in place.
        """
        kaiming_normal_ = _safe_import("torch.nn.init.kaiming_normal_")

        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                kaiming_normal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor containing the time series data, of shape
            ``(batch_size, seq_len, input_size)`` if ``batch_first``, else
            ``(seq_len, batch_size, input_size)``.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Class scores. Raw outputs, i.e., logits, unless ``activation``
            is passed.
        """
        if isinstance(X, np.ndarray):
            from_numpy = self._torch_op("torch.from_numpy")
            X = from_numpy(X).float()

        # both branches consume the time axis, which is 1 for batch-first
        # inputs and 0 otherwise
        gru_out, _ = self.gru(X)
        if self.batch_first:
            # GRU branch, keeping the last hidden state
            gru_out = gru_out[:, -1, :]
            # (batch, time, channels) -> (batch, channels, time) for Conv1d
            conv_out = X.permute(0, 2, 1)
        else:
            gru_out = gru_out[-1, :, :]
            # (time, batch, channels) -> (batch, channels, time) for Conv1d
            conv_out = X.permute(1, 2, 0)
        gru_out = self.grudropout(gru_out)

        # fully convolutional branch, globally average pooled
        for conv_block in self.conv_blocks:
            conv_out = conv_block(conv_out)
        conv_out = self.globalavgpool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)

        cat = self._torch_op("torch.cat")
        out = cat((gru_out, conv_out), dim=1)
        out = self.fc(out)
        if self.activation is not None:
            out = self.activation(out)
        return out
