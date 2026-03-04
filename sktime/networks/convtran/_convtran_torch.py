"""ConvTran neural network architecture in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["ConvTranNetworkTorch"]

import numpy as np

from sktime.networks.convtran._attention import (
    Attention,
    AttentionRelScalar,
    AttentionRelVector,
)
from sktime.networks.convtran._positional_encoding import (
    AbsolutePositionalEncoding,
    LearnablePositionalEncoding,
    tAPE,
)
from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")
Conv1d = _safe_import("torch.nn.Conv1d")  # base class for CausalConv1d


class ConvTranNetworkTorch(NNModule):
    """Establish the network structure for ConvTran in PyTorch.

    Adapted from the implementation used in [1]_.

    Parameters
    ----------
    input_size : int or tuple of int
        Number of expected features in the input. If tuple, must be of length 3
        and in format (n_instances, n_dims, series_length).
    num_classes : int
        Number of outputs.
    net_type : str, default="C-T"
        Network type to use. Should be one of "T" (Transformer),
        "C-T" (ConvTran) or "C-CT" (Causal ConvTran).
    activation : str or None, default=None
        Activation function to use in the output layer.
    activation_hidden : str, default="relu"
        Activation function to use in the hidden layers.
    emb_size : int, default=16
        Embedding dimension used in attention and feed-forward blocks.
    dim_ff : int, default=256
        Hidden dimension of the feed-forward block.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.01
        Dropout rate applied in attention and feed-forward blocks.
    use_abs_pos_encoding : bool, default=True
        Whether to apply absolute positional encoding.
    use_rel_pos_encoding : bool, default=True
        Whether to apply relative positional encoding.
    abs_pos_encoding_scheme : str or None, default="tAPE"
        Absolute positional encoding scheme. Supported values:
        "tAPE", "sin", "learn", or None.
    rel_pos_encoding_scheme : str or None, default="erpe"
        Relative positional encoding scheme. Supported values:
        "erpe", "vector", or None.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
        Mahsa Salehi. Improving position encoding of transformers for multivariate
        time series classification. Data Mining and Knowledge Discovery,
        38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
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
        net_type: str = "C-T",
        activation: str | None = None,
        activation_hidden: str = "relu",
        emb_size: int = 16,
        dim_ff: int = 256,
        num_heads: int = 8,
        dropout: float = 0.01,
        use_abs_pos_encoding: bool = True,
        use_rel_pos_encoding: bool = True,
        abs_pos_encoding_scheme: str | None = "tAPE",
        rel_pos_encoding_scheme: str | None = "erpe",
    ):
        super().__init__()

        self._import_cache = {}
        self.input_size = input_size
        self.num_classes = num_classes
        self.net_type = net_type
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.emb_size = emb_size
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_abs_pos_encoding = use_abs_pos_encoding
        self.use_rel_pos_encoding = use_rel_pos_encoding
        self.abs_pos_encoding_scheme = abs_pos_encoding_scheme
        self.rel_pos_encoding_scheme = rel_pos_encoding_scheme

        net_type = self.net_type.upper().replace("_", "-")
        if net_type == "CC-T":
            net_type = "C-CT"
        if net_type not in {"T", "C-T", "C-CT"}:
            raise ValueError("`net_type` must be one of {'T', 'C-T', 'C-CT'}.")
        self.net_type = net_type

        if isinstance(self.input_size, int):
            n_dims = self.input_size
            seq_len = None
        elif isinstance(self.input_size, tuple):
            if len(self.input_size) == 3:
                _, n_dims, seq_len = self.input_size
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
        self.seq_len = seq_len

        abs_scheme = self._normalize_scheme(self.abs_pos_encoding_scheme)
        rel_scheme = self._normalize_scheme(self.rel_pos_encoding_scheme)
        if not self.use_abs_pos_encoding:
            abs_scheme = None
        if not self.use_rel_pos_encoding:
            rel_scheme = None
        allowed_abs = {None, "tape", "sin", "learn"}
        allowed_rel = {None, "erpe", "vector"}
        if abs_scheme not in allowed_abs:
            raise ValueError(
                "`abs_pos_encoding_scheme` must be one of "
                "{'tAPE', 'sin', 'learn', None}."
            )
        if rel_scheme not in allowed_rel:
            raise ValueError(
                "`rel_pos_encoding_scheme` must be one of {'erpe', 'vector', None}."
            )
        self._abs_scheme = abs_scheme
        self._rel_scheme = rel_scheme

        if self.seq_len is None and (self._abs_scheme or self._rel_scheme):
            # that is, input_dim should NOT be just an int,
            # it should be a tuple of length 3
            raise ValueError(
                "`input_size` must provide series_length when positional encoding "
                "is enabled."
            )

        self._activation_hidden = self._instantiate_hidden_activation()
        self._activation_out = (
            self._instantiate_output_activation() if self.activation else None
        )

        self._build_blocks()

    def _torch_op(self, import_path):
        """Lazy import and cache torch ops used in forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _normalize_scheme(self, scheme):
        if scheme is None:
            return None
        if not isinstance(scheme, str):
            raise TypeError("Encoding scheme must be a string or None.")
        scheme = scheme.strip().lower()
        if scheme in {"none", "null"}:
            return None
        return scheme

    def _build_blocks(self):
        if self.net_type == "T":
            self._build_transformer_blocks()
        elif self.net_type == "C-T":
            self._build_convtran_blocks()
        else:
            self._build_causal_convtran_blocks()

        nnSequential = _safe_import("torch.nn.Sequential")
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")

        self.feed_forward = nnSequential(
            nnLinear(self.emb_size, self.dim_ff),
            self._activation_hidden,
            nnDropout(self.dropout),
            nnLinear(self.dim_ff, self.emb_size),
            nnDropout(self.dropout),
        )

        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnFlatten = _safe_import("torch.nn.Flatten")
        self.gap = nnAdaptiveAvgPool1d(1)
        self.flatten = nnFlatten()

        self.out = nnLinear(self.emb_size, self.num_classes)

    def _build_transformer_blocks(self):
        nnSequential = _safe_import("torch.nn.Sequential")
        nnLinear = _safe_import("torch.nn.Linear")
        nnLayerNorm = _safe_import("torch.nn.LayerNorm")

        self.embed_layer = nnSequential(
            nnLinear(self.n_dims, self.emb_size),
            nnLayerNorm(self.emb_size, eps=1e-5),
        )

        if self._abs_scheme == "tape":
            self.fix_position = tAPE(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        elif self._abs_scheme == "sin":
            self.fix_position = AbsolutePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        elif self._abs_scheme == "learn":
            self.fix_position = LearnablePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        else:
            self.fix_position = None

        if self._rel_scheme == "erpe":
            self.attention_layer = AttentionRelScalar(
                self.emb_size, self.num_heads, self.seq_len, self.dropout
            )
        elif self._rel_scheme == "vector":
            self.attention_layer = AttentionRelVector(
                self.emb_size, self.num_heads, self.seq_len, self.dropout
            )
        else:
            self.attention_layer = Attention(
                self.emb_size, self.num_heads, self.dropout
            )

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.layer_norm_1 = nnLayerNorm(self.emb_size, eps=1e-5)
        self.layer_norm_2 = nnLayerNorm(self.emb_size, eps=1e-5)

    def _build_convtran_blocks(self):
        nnSequential = _safe_import("torch.nn.Sequential")
        nnConv2d = _safe_import("torch.nn.Conv2d")
        nnBatchNorm2d = _safe_import("torch.nn.BatchNorm2d")
        nnGELU = _safe_import("torch.nn.GELU")

        self.embed_layer = nnSequential(
            nnConv2d(1, self.emb_size * 4, kernel_size=(1, 8), padding="same"),
            nnBatchNorm2d(self.emb_size * 4),
            nnGELU(),
        )
        self.embed_layer2 = nnSequential(
            nnConv2d(
                self.emb_size * 4,
                self.emb_size,
                kernel_size=(self.n_dims, 1),
                padding="valid",
            ),
            nnBatchNorm2d(self.emb_size),
            nnGELU(),
        )

        if self._abs_scheme == "tape":
            self.fix_position = tAPE(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        elif self._abs_scheme == "sin":
            self.fix_position = AbsolutePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        elif self._abs_scheme == "learn":
            self.fix_position = LearnablePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len
            )
        else:
            self.fix_position = None

        if self._rel_scheme == "erpe":
            self.attention_layer = AttentionRelScalar(
                self.emb_size, self.num_heads, self.seq_len, self.dropout
            )
        elif self._rel_scheme == "vector":
            self.attention_layer = AttentionRelVector(
                self.emb_size, self.num_heads, self.seq_len, self.dropout
            )
        else:
            self.attention_layer = Attention(
                self.emb_size, self.num_heads, self.dropout
            )

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.layer_norm_1 = nnLayerNorm(self.emb_size, eps=1e-5)
        self.layer_norm_2 = nnLayerNorm(self.emb_size, eps=1e-5)

    def _build_causal_convtran_blocks(self):
        nnSequential = _safe_import("torch.nn.Sequential")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnGELU = _safe_import("torch.nn.GELU")

        self.causal_conv1 = nnSequential(
            CausalConv1d(
                self.n_dims,
                self.emb_size,
                kernel_size=8,
                stride=2,
                dilation=1,
            ),
            nnBatchNorm1d(self.emb_size),
            nnGELU(),
        )
        self.causal_conv2 = nnSequential(
            CausalConv1d(
                self.emb_size,
                self.emb_size,
                kernel_size=5,
                stride=2,
                dilation=2,
            ),
            nnBatchNorm1d(self.emb_size),
            nnGELU(),
        )
        self.causal_conv3 = nnSequential(
            CausalConv1d(
                self.emb_size,
                self.emb_size,
                kernel_size=3,
                stride=2,
                dilation=2,
            ),
            nnBatchNorm1d(self.emb_size),
            nnGELU(),
        )

        seq_len = self.seq_len
        if seq_len is None:
            raise ValueError("`input_size` must provide series_length for C-CT.")
        seq_len = self._causal_out_len(seq_len, kernel_size=8, stride=2, dilation=1)
        seq_len = self._causal_out_len(seq_len, kernel_size=5, stride=2, dilation=2)
        seq_len = self._causal_out_len(seq_len, kernel_size=3, stride=2, dilation=2)
        self.seq_len_after_conv = seq_len

        if self._abs_scheme == "tape":
            self.fix_position = tAPE(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len_after_conv
            )
        elif self._abs_scheme == "sin":
            self.fix_position = AbsolutePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len_after_conv
            )
        elif self._abs_scheme == "learn":
            self.fix_position = LearnablePositionalEncoding(
                self.emb_size, dropout=self.dropout, max_len=self.seq_len_after_conv
            )
        else:
            self.fix_position = None

        if self._rel_scheme == "erpe":
            self.attention_layer = AttentionRelScalar(
                self.emb_size, self.num_heads, self.seq_len_after_conv, self.dropout
            )
        elif self._rel_scheme == "vector":
            self.attention_layer = AttentionRelVector(
                self.emb_size, self.num_heads, self.seq_len_after_conv, self.dropout
            )
        else:
            self.attention_layer = Attention(
                self.emb_size, self.num_heads, self.dropout
            )

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.layer_norm_1 = nnLayerNorm(self.emb_size, eps=1e-5)
        self.layer_norm_2 = nnLayerNorm(self.emb_size, eps=1e-5)

    def _instantiate_hidden_activation(self):
        if isinstance(self.activation_hidden, NNModule):
            return self.activation_hidden
        if not isinstance(self.activation_hidden, str):
            raise TypeError(
                "`activation_hidden` should be a str or torch.nn.Module. "
                f"Found {type(self.activation_hidden)}"
            )

        act = self.activation_hidden.lower()
        if act == "relu":
            return _safe_import("torch.nn.ReLU")()
        if act in {"leaky_relu", "leakyrelu"}:
            return _safe_import("torch.nn.LeakyReLU")(negative_slope=0.01)
        if act == "gelu":
            return _safe_import("torch.nn.GELU")()

        raise ValueError(
            "Unsupported activation_hidden. Supported: "
            "'relu', 'leaky_relu', 'gelu'. "
            f"Found {self.activation_hidden}"
        )

    def _instantiate_output_activation(self):
        if isinstance(self.activation, NNModule):
            return self.activation
        if not isinstance(self.activation, str):
            raise TypeError(
                "`activation` should be a str or torch.nn.Module. "
                f"Found {type(self.activation)}"
            )

        act = self.activation.lower()
        if act == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        if act == "softmax":
            return _safe_import("torch.nn.Softmax")(dim=1)
        if act == "logsoftmax":
            return _safe_import("torch.nn.LogSoftmax")(dim=1)
        if act == "logsigmoid":
            return _safe_import("torch.nn.LogSigmoid")()

        raise ValueError(
            "Unsupported activation. Supported: "
            "'sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'. "
            f"Found {self.activation}"
        )

    def _causal_out_len(self, length, kernel_size, stride, dilation):
        pad = (kernel_size - 1) * dilation
        return (length + pad - dilation * (kernel_size - 1) - 1) // stride + 1

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

        if X.shape[1] == self.n_dims and X.shape[2] != self.n_dims:
            # (batch, n_dims, seq_len) -> (batch, seq_len, n_dims)
            X = X.transpose(1, 2)

        if self.net_type == "T":
            return self._forward_transformer(X)
        if self.net_type == "C-T":
            return self._forward_convtran(X)
        return self._forward_causal_convtran(X)

    def _forward_transformer(self, X):
        x_src = self.embed_layer(X)
        if self.fix_position is not None:
            x_src = self.fix_position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.layer_norm_1(att)
        out = att + self.feed_forward(att)
        out = self.layer_norm_2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        if self._activation_out is not None:
            out = self._activation_out(out)
        return out

    def _forward_convtran(self, X):
        # (batch, seq_len, n_dims) -> (batch, n_dims, seq_len) for 2D conv
        x = X.transpose(1, 2)
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.fix_position is not None:
            x_src_pos = self.fix_position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.layer_norm_1(att)
        out = att + self.feed_forward(att)
        out = self.layer_norm_2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        if self._activation_out is not None:
            out = self._activation_out(out)
        return out

    def _forward_causal_convtran(self, X):
        # (batch, seq_len, n_dims) -> (batch, n_dims, seq_len) for 1D conv
        x = X.transpose(1, 2)
        x_src = self.causal_conv1(x)
        x_src = self.causal_conv2(x_src)
        x_src = self.causal_conv3(x_src)
        x_src = x_src.permute(0, 2, 1)
        if self.fix_position is not None:
            x_src_pos = self.fix_position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.layer_norm_1(att)
        out = att + self.feed_forward(att)
        out = self.layer_norm_2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        if self._activation_out is not None:
            out = self._activation_out(out)
        return out


class CausalConv1d(Conv1d):
    """Causal 1D convolution with left padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self._import_cache = {}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.__padding = (kernel_size - 1) * dilation

    def _torch_op(self, import_path):
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def forward(self, x):
        pad = self._torch_op("torch.nn.functional.pad")
        return super().forward(pad(x, (self.__padding, 0)))
