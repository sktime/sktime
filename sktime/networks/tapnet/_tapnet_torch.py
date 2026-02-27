"""TapNet Neural Network for Classification and Regression in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["TapNetNetworkTorch"]

import math
import warnings

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class TapNetNetworkTorch(NNModule):
    """Establish the network structure for TapNet in PyTorch.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    input_size : int or tuple of int
        Number of expected features in the input. If tuple, must be of length 3
        and in format (n_instances, n_dims, series_length).
    num_classes : int
        Number of outputs.
    activation : str or None, default=None
        Activation function to use in the output layer.
    activation_hidden : str, default="leaky_relu"
        Activation function to use in the hidden layers.
    kernel_size : tuple of int, default = (8, 5, 3)
        Specifying the length of the 1D convolution window.
    layers : tuple of int, default = (500, 300)
        Size of dense layers in the mapping section.
    filter_sizes : tuple of int, default = (256, 256, 128)
        Number of convolutional filters in each conv block.
        If ``use_rp`` is True, the first conv layer is group-specific and all
        subsequent conv layers share parameters across groups.
    dropout : float, default = 0.5
        Dropout rate for the convolutional layers.
    lstm_dropout : float, default = 0.8
        Dropout rate for the LSTM layer.
    dilation : int, default = 1
        Dilation value.
    padding : str, default = "same"
        Type of padding for convolution layers.
    use_rp : bool, default = True
        Whether to use random projections.
    rp_group : int, default = 3
        Number of random permutation groups g for random dimension permutation (RDP).
        Must be a positive integer.
    rp_alpha : float, default = 2.0
        Scale factor alpha used to compute the RDP group size:
        rp_dim = floor(n_dims * rp_alpha / rp_group).
        If rp_dim becomes 0, RDP is disabled with a warning (RDP requires
        multivariate inputs).
        Must be positive.
    use_att : bool, default = True
        Whether to use self attention.
    use_lstm : bool, default = True
        Whether to use an LSTM layer.
    use_cnn : bool, default = True
        Whether to use a CNN layer.
    random_state : int or None, default=None
        Seed to ensure reproducibility.
    init_weights : bool, default = True
        Whether to apply custom initialization.
    fc_dropout : float, default = 0.0
        Dropout rate before the output layer.

    References
    ----------
    .. [1] Zhang et al. Tapnet: Multivariate time series classification with
    attentional prototypical network,
    Proceedings of the AAAI Conference on Artificial Intelligence
    34(4), 6845-6852, 2020
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
        activation: str | None = None,
        activation_hidden: str = "leaky_relu",
        kernel_size: tuple[int, ...] = (8, 5, 3),
        layers: tuple[int, ...] = (500, 300),
        filter_sizes: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.5,
        lstm_dropout: float = 0.8,
        dilation: int = 1,
        padding: str = "same",
        use_rp: bool = True,
        rp_group: int = 3,
        rp_alpha: float = 2.0,
        use_att: bool = True,
        use_lstm: bool = True,
        use_cnn: bool = True,
        random_state: int | None = None,
        init_weights: bool = True,
        fc_dropout: float = 0.0,
    ):
        super().__init__()

        self._import_cache = {}
        self.input_size = input_size
        self.num_classes = num_classes
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.kernel_size = kernel_size
        self.layers = layers
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.dilation = dilation
        self.padding = padding
        self.use_rp = use_rp
        self.rp_group = rp_group
        self.rp_alpha = rp_alpha
        self.use_att = use_att
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.random_state = random_state
        self.init_weights = init_weights
        self.fc_dropout = fc_dropout

        if not isinstance(self.kernel_size, tuple) or not all(
            isinstance(k, int) for k in self.kernel_size
        ):
            raise TypeError("`kernel_size` must be a tuple of ints.")

        if not isinstance(self.filter_sizes, tuple) or not all(
            isinstance(f, int) for f in self.filter_sizes
        ):
            raise TypeError("`filter_sizes` must be a tuple of ints.")

        if len(self.kernel_size) != len(self.filter_sizes):
            raise ValueError(
                "`kernel_size` and `filter_sizes` must be of the same length."
            )
        if len(self.kernel_size) < 1:
            raise ValueError("`kernel_size` and `filter_sizes` must have length >= 1.")

        if not isinstance(self.layers, tuple) or not all(
            isinstance(l, int) for l in self.layers
        ):
            raise TypeError("`layers` must be a tuple of ints.")

        if not isinstance(self.rp_group, int) or self.rp_group < 1:
            raise ValueError("`rp_group` must be a positive integer.")
        if not isinstance(self.rp_alpha, (int, float)) or self.rp_alpha <= 0:
            raise ValueError("`rp_alpha` must be a positive number.")

        # Validate input size and infer n_dims
        if isinstance(self.input_size, int):
            n_dims = self.input_size
            series_length = None
        elif isinstance(self.input_size, tuple):
            if len(self.input_size) == 3:
                _, n_dims, series_length = self.input_size
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

        # RNG setup for reproducibility
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)
            torch_manual_seed = _safe_import("torch.manual_seed")
            torch_manual_seed(self.random_state)
        else:
            self._rng = np.random.default_rng()

        # Activations
        self._activation_hidden = self._instantiate_hidden_activation()
        self._activation_out = (
            self._instantiate_output_activation() if self.activation else None
        )

        # Random projection parameters (RDP)
        if self.use_rp:
            self.rp_dim = math.floor(self.n_dims * self.rp_alpha / self.rp_group)
            if self.rp_dim < 1:
                warnings.warn(
                    "Disabling random dimension permutation (RDP) because the "
                    "computed rp_dim is 0. This can happen for univariate data; "
                    "RDP requires multivariate inputs.",
                    UserWarning,
                )
                self.use_rp = False
                self.rp_dim = 0
            else:
                # rp_dim cannot be greater than n_dims
                self.rp_dim = min(self.rp_dim, self.n_dims)
        else:
            self.rp_dim = 0

        # Layers for LSTM
        if self.use_lstm:
            self.lstm_dim = 128
            LSTM = _safe_import("torch.nn.LSTM")
            self.lstm = LSTM(
                input_size=self.n_dims,
                hidden_size=self.lstm_dim,
                batch_first=True,
            )
            Dropout = _safe_import("torch.nn.Dropout")
            self.lstm_dropout_layer = Dropout(p=self.lstm_dropout)
            if self.use_att:
                SeqSelfAttentionTorch = _safe_import(
                    "sktime.libs._torch_self_attention.seq_self_attention.SeqSelfAttentionTorch"
                )
                self.lstm_attn = SeqSelfAttentionTorch(
                    self.lstm_dim,
                    attention_type="multiplicative",
                    input_dim=self.lstm_dim,
                )

        # Layers for CNN
        if self.use_cnn:
            if self.use_rp:
                ModuleList = _safe_import("torch.nn.ModuleList")
                self.rp_conv1 = ModuleList(
                    [self._make_conv1_block(self.rp_dim) for _ in range(self.rp_group)]
                )
                self.rp_conv_shared = self._make_conv_shared_block()
                if self.use_att:
                    SeqSelfAttentionTorch = _safe_import(
                        "sktime.libs._torch_self_attention.seq_self_attention.SeqSelfAttentionTorch"
                    )
                    self.rp_attn = SeqSelfAttentionTorch(
                        self.filter_sizes[-1],
                        attention_type="multiplicative",
                        input_dim=self.filter_sizes[-1],
                    )
                # Pre-compute RP indices (deterministic if random_state is set)
                for i in range(self.rp_group):
                    idx = self._rng.permutation(self.n_dims)[: self.rp_dim]
                    torch_tensor = _safe_import("torch.tensor")
                    torch_long = _safe_import("torch.long")
                    self.register_buffer(
                        f"rp_idx_{i}",
                        torch_tensor(idx, dtype=torch_long),
                    )
            else:
                self.conv1 = self._make_conv1_block(self.n_dims)
                self.conv_shared = self._make_conv_shared_block()
                if self.use_att:
                    SeqSelfAttentionTorch = _safe_import(
                        "sktime.libs._torch_self_attention.seq_self_attention.SeqSelfAttentionTorch"
                    )
                    self.cnn_attn = SeqSelfAttentionTorch(
                        self.filter_sizes[-1],
                        input_dim=self.filter_sizes[-1],
                    )

        if not self.use_lstm and not self.use_cnn:
            raise ValueError("At least one of `use_lstm` or `use_cnn` must be True.")

        # Mapping head
        mapping_in = 0
        if self.use_cnn:
            mapping_in += self.filter_sizes[-1] * (self.rp_group if self.use_rp else 1)
        if self.use_lstm:
            mapping_in += self.lstm_dim

        Linear = _safe_import("torch.nn.Linear")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        self.fc1 = Linear(mapping_in, self.layers[0])
        self.bn1 = BatchNorm1d(self.layers[0])
        self.fc2 = Linear(self.layers[0], self.layers[1])

        if self.fc_dropout:
            Dropout = _safe_import("torch.nn.Dropout")
            self.out_dropout = Dropout(p=self.fc_dropout)

        self.out = Linear(self.layers[1], self.num_classes)

        if self.init_weights:
            self.apply(self._init_weights)

    def _torch_op(self, import_path):
        """Lazy import and cache torch ops used in forward pass."""
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _make_conv1_block(self, in_channels):
        """Build the first 1D CNN block in TapNet."""
        Sequential = _safe_import("torch.nn.Sequential")
        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        return Sequential(
            Conv1d(
                in_channels,
                self.filter_sizes[0],
                kernel_size=self.kernel_size[0],
                dilation=self.dilation,
                padding=self.padding,
            ),
            BatchNorm1d(self.filter_sizes[0]),
            self._instantiate_hidden_activation(),
        )

    def _make_conv_shared_block(self):
        """Build shared CNN blocks for layers after the first."""
        if len(self.filter_sizes) < 2:
            return None
        Sequential = _safe_import("torch.nn.Sequential")
        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        layers = []
        in_channels = self.filter_sizes[0]
        for i in range(1, len(self.filter_sizes)):
            layers.append(
                Conv1d(
                    in_channels,
                    self.filter_sizes[i],
                    kernel_size=self.kernel_size[i],
                    dilation=self.dilation,
                    padding=self.padding,
                )
            )
            layers.append(BatchNorm1d(self.filter_sizes[i]))
            layers.append(self._instantiate_hidden_activation())
            in_channels = self.filter_sizes[i]
        return Sequential(*layers)

    def _instantiate_hidden_activation(self):
        """Instantiate the activation function to be applied on the hidden layers.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the hidden layers.
        """
        if isinstance(self.activation_hidden, NNModule):
            return self.activation_hidden
        if not isinstance(self.activation_hidden, str):
            raise TypeError(
                "`activation_hidden` should be a str or torch.nn.Module. "
                f"Found {type(self.activation_hidden)}"
            )

        act = self.activation_hidden.lower()
        if act in ("relu",):
            return _safe_import("torch.nn.ReLU")()
        if act in ("leaky_relu", "leakyrelu"):
            return _safe_import("torch.nn.LeakyReLU")(negative_slope=0.01)

        raise ValueError(
            "Unsupported activation_hidden. Supported: "
            "'relu', 'leaky_relu'. "
            f"Found {self.activation_hidden}"
        )

    def _instantiate_output_activation(self):
        """Instantiate the activation function to be applied on the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
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

    def _init_weights(self, module):
        """Apply tensorflow-like initializations.

        Parameters
        ----------
        module : torch.nn.Module
            Input module on which to apply the initialization.
        """
        Conv1d = _safe_import("torch.nn.Conv1d")
        Linear = _safe_import("torch.nn.Linear")
        LSTM = _safe_import("torch.nn.LSTM")
        xavier_uniform_ = _safe_import("torch.nn.init.xavier_uniform_")
        orthogonal_ = _safe_import("torch.nn.init.orthogonal_")
        zeros_ = _safe_import("torch.nn.init.zeros_")
        if isinstance(module, Conv1d) or isinstance(module, Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)

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

        if self.use_lstm:
            x_lstm, _ = self.lstm(X)
            x_lstm = self.lstm_dropout_layer(x_lstm)
            if self.use_att:
                x_lstm = self.lstm_attn(x_lstm)
            x_lstm = x_lstm.mean(dim=1)  # global average pool over time dimension

        if self.use_cnn:
            x_cnn = X.transpose(1, 2)
            if self.use_rp:
                rp_outs = []
                for i in range(self.rp_group):
                    idx = getattr(self, f"rp_idx_{i}")
                    x_proj = x_cnn.index_select(dim=1, index=idx)
                    x_conv = self.rp_conv1[i](x_proj)
                    if self.rp_conv_shared is not None:
                        x_conv = self.rp_conv_shared(x_conv)
                    if self.use_att:
                        x_conv = self.rp_attn(x_conv.transpose(1, 2))
                        x_conv = x_conv.transpose(1, 2)
                    x_conv = x_conv.mean(dim=2)
                    rp_outs.append(x_conv)
                torch_cat = self._torch_op("torch.cat")
                x_cnn = torch_cat(rp_outs, dim=1)  # concatenate all rp group vectors
            else:
                x_conv = self.conv1(x_cnn)
                if self.conv_shared is not None:
                    x_conv = self.conv_shared(x_conv)
                if self.use_att:
                    x_conv = self.cnn_attn(x_conv.transpose(1, 2))
                    x_conv = x_conv.transpose(1, 2)
                x_cnn = x_conv.mean(dim=2)

        if self.use_lstm and self.use_cnn:
            torch_cat = self._torch_op("torch.cat")
            x = torch_cat([x_cnn, x_lstm], dim=1)
        elif self.use_lstm:
            x = x_lstm
        else:
            x = x_cnn

        x = self.fc1(x)
        x = self._activation_hidden(x)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.fc2(x)
        if self.fc_dropout:
            x = self.out_dropout(x)
        x = self.out(x)
        if self._activation_out is not None:
            x = self._activation_out(x)
        return x
