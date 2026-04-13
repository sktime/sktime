"""CNTC for Classification and Regression in PyTorch."""

__authors__ = ["fnhirwa"]
__all__ = ["CNTCNetworkTorch"]


from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
torch = _safe_import("torch")
nn = _safe_import("torch.nn")
torchF = _safe_import("torch.nn.functional")
NNModule = _safe_import("torch.nn.Module")
Dataset = _safe_import("torch.utils.data.Dataset")


class _CNTCDataset(Dataset):
    """Dataset that returns dict-based inputs for CNTCNetworkTorch.

    Wraps two tensors (x1 and x3) and an optional label tensor, returning
    items as ``({"x1": ..., "x3": ...}, y)`` tuples so that the base-class
    ``_run_epoch`` can unpack them and call ``network(**inputs)`` as
    ``network(x1=..., x3=...)``.
    """

    def __init__(self, X1, X3, y=None):
        self.X1 = X1
        self.X3 = X3
        self.y = y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        inputs = {"x1": self.X1[idx], "x3": self.X3[idx]}
        if self.y is not None:
            return inputs, self.y[idx]
        return inputs


class SeqSelfAttention(NNModule):
    """Additive self-attention over a sequence.

    Input / Output: [B, T, C]
    """

    def __init__(
        self,
        embed_dim: int,
        attention_width: int = 10,
        activation_attention: str = "sigmoid",
    ):
        super().__init__()
        self.attention_width = attention_width
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)
        self._act = getattr(torchF, activation_attention)

    def forward(self, X):
        q = self.W_q(X)
        k = self.W_k(X)
        score = self.v(torch.tanh(q.unsqueeze(2) + k.unsqueeze(1))).squeeze(-1)
        if self.attention_width is not None:
            T = X.size(1)
            mask = torch.tril(
                torch.ones(T, T, device=X.device), self.attention_width - 1
            )
            mask = torch.triu(mask, -(self.attention_width - 1))
            score = score.masked_fill(mask == 0, float("-inf"))
        attn = self._act(score)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        return torch.bmm(attn, X)


class CNTCNetworkTorch(NNModule):
    """CNTC (Contextual Neural Networks for Time Series Classification).

    CNTC is a deep learning network architecture for time series classification.
    It combines two parallel arms — a Contextual CNN (CCNN) and a Contextual LSTM
    (CLSTM) — fused via concatenation, followed by self-attention and an MLP head.

    Architecture summary:
            - Arm 1+2 (CCNN): Conv1D on raw input + SimpleRNN context, merged on the
                time axis, then zero or more additional Conv1D blocks.
      - Arm 3 (CLSTM): LSTM on rolling-mean-augmented input.
      - Head: Merged sequence → MaxPool → Self-Attention → MLP → Linear output.

    For more details on the architecture, see [1]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels (feature dimension per time step).
        For univariate time series this is 1.
    n_classes : int
        Number of output classes for classification.
    kernel_sizes : tuple of int, length n_conv_layers, default=(1, 1)
        Kernel size for each Conv1D layer.
    rnn_layer : int, default=64
        Hidden size of the SimpleRNN used in the CCNN arm.
    lstm_layer : int, default=8
        Hidden size of the LSTM used in the CLSTM arm.
    avg_pool_size : int, default=1
        Kernel size of the MaxPool1D layer applied after merging all arms.
        The original paper uses 1 (no-op pooling).
    n_conv_layers : int, default=2
        Number of Conv1D blocks in the CCNN arm (must match len(kernel_sizes)
        and len(filter_sizes)).
    filter_sizes : tuple of int, length n_conv_layers, default=(16, 8)
        Number of output filters for each Conv1D block.
    dense_size : int, default=64
        Number of units in each of the two MLP hidden layers.
    activation : str, default='relu'
        Activation function name for hidden layers (excluding attention).
        Must be a valid attribute of torch.nn.functional, e.g. 'relu', 'tanh'.
    activation_attention : str, default='sigmoid'
        Activation function name applied inside the self-attention score
        computation. Must be a valid attribute of torch.nn.functional.
    dropout : float or dict or tuple or None, default=None
        Dropout rate(s) applied in the network. If a single float, the same
        rate is applied everywhere. If a dict, it specifies dropout rates
        for individual network components. For the two-conv architecture as
        specified in the paper 7-tuple value dropout is supported.

        Valid keys include: 'conv', 'rnn', 'lstm', 'pool', 'attention', 'mlp'.
        - The 'conv' key can map to a single float (applied to all conv layers)
        or a sequence of floats of length `n_conv_layers`.
        - Unspecified keys will inherit the default CNTC dropout profile.

        Example: {'conv': [0.8, 0.7], 'rnn': 0.8, 'mlp': 0.8}
    init_weights: str, default="xavier_uniform"
        Weight initialization method for all layers. Must be a valid method in
        torch.nn.init, e.g. 'xavier_uniform', 'kaiming_normal', etc.
    random_state : int, default=0
        Seed for reproducible weight initialisation.

    References
    ----------
    .. [1] Kamara, A.F., Chen, E., Liu, Q., Pan, Z. (2020).
        Combining contextual neural networks for time series classification.
        Neurocomputing, 384, 57-66.
        https://doi.org/10.1016/j.neucom.2019.10.113
    """

    _tags = {
        # packaging info
        # --------------
        "authors": __authors__,
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    _DROPOUT_COMPONENTS = ("rnn", "lstm", "pool", "attention", "mlp")

    @staticmethod
    def _check_dropout(name, value):
        """Validate a dropout rate and return it as a float."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Dropout for '{name}' must be a float")

        value = float(value)
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"Dropout for '{name}' must be in the range [0, 1), got {value}"
            )
        return value

    @classmethod
    def _default_dropout_config(cls, n_conv_layers):
        """Construct the default dropout profile for the configured conv stack."""
        conv_dropout = [0.8]
        if n_conv_layers > 1:
            conv_dropout.extend([0.7] * (n_conv_layers - 1))

        return {
            "conv": conv_dropout,
            "rnn": 0.8,
            "lstm": 0.8,
            "pool": 0.6,
            "attention": 0.5,
            "mlp": 0.8,
        }

    @classmethod
    def _map_dropout(cls, dropout, n_conv_layers):
        """Normalize supported dropout formats to a dict keyed by component."""
        normalized = cls._default_dropout_config(n_conv_layers)

        if dropout is None:
            return normalized

        if isinstance(dropout, (int, float)):
            rate = cls._check_dropout("dropout", dropout)
            normalized["conv"] = [rate] * n_conv_layers
            for key in cls._DROPOUT_COMPONENTS:
                normalized[key] = rate
            return normalized

        if isinstance(dropout, (list, tuple)):
            if len(dropout) != 7 or n_conv_layers != 2:
                raise ValueError(
                    "Tuple dropout is only supported for the two-conv "
                    "architecture. Use a dict with a 'conv' sequence for "
                    "custom conv stacks."
                )

            conv1, rnn, conv2, lstm, pool, attention, mlp = dropout
            return {
                "conv": [
                    cls._check_dropout("conv[0]", conv1),
                    cls._check_dropout("conv[1]", conv2),
                ],
                "rnn": cls._check_dropout("rnn", rnn),
                "lstm": cls._check_dropout("lstm", lstm),
                "pool": cls._check_dropout("pool", pool),
                "attention": cls._check_dropout("attention", attention),
                "mlp": cls._check_dropout("mlp", mlp),
            }

        if not isinstance(dropout, dict):
            raise ValueError(
                "Dropout must be None, a float, a dict, or a legacy 7-value tuple"
            )

        valid_keys = {"conv", *cls._DROPOUT_COMPONENTS}
        for key, value in dropout.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid dropout key: {key}. Valid keys are: {valid_keys}"
                )

            if key == "conv":
                if isinstance(value, (int, float)):
                    rate = cls._check_dropout("conv", value)
                    normalized["conv"] = [rate] * n_conv_layers
                elif isinstance(value, (list, tuple)):
                    if len(value) != n_conv_layers:
                        raise ValueError(
                            f"Dropout for 'conv' must have length "
                            f"{n_conv_layers}, got {len(value)}"
                        )
                    normalized["conv"] = [
                        cls._check_dropout(f"conv[{i}]", rate)
                        for i, rate in enumerate(value)
                    ]
                else:
                    raise ValueError(
                        "Dropout for 'conv' must be a float or a list/tuple of floats"
                    )
            else:
                normalized[key] = cls._check_dropout(key, value)

        return normalized

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        kernel_sizes: tuple = (1, 1),
        rnn_layer: int = 64,
        lstm_layer: int = 8,
        avg_pool_size: int = 1,
        n_conv_layers: int = 2,
        filter_sizes: tuple = (16, 8),
        dense_size: int = 64,
        activation: str = "relu",
        activation_attention: str = "sigmoid",
        dropout: float | dict | tuple | None = None,
        init_weights: str | None = "xavier_uniform",
        random_state: int = 0,
    ):
        super().__init__()

        if n_conv_layers < 1:
            raise ValueError("n_conv_layers must be at least 1")

        if len(kernel_sizes) != n_conv_layers:
            raise ValueError(
                f"len(kernel_sizes)={len(kernel_sizes)} "
                f"must equal n_conv_layers={n_conv_layers}"
            )
        if len(filter_sizes) != n_conv_layers:
            raise ValueError(
                f"len(filter_sizes)={len(filter_sizes)} "
                f"must equal n_conv_layers={n_conv_layers}"
            )

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.activation = activation
        self.activation_attention = activation_attention
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.dense_size = dense_size
        self.random_state = random_state
        self.rnn_layer = rnn_layer
        self.lstm_layer = lstm_layer
        self.init_weights = init_weights
        self.dropout = self._map_dropout(dropout, n_conv_layers)

        self._act = getattr(torchF, activation)
        rnn_nonlinearity = "tanh" if activation == "tanh" else "relu"
        self.rnn_nonlinearity = rnn_nonlinearity

        torch.manual_seed(random_state)

        # arm 1: first conv block
        self.conv1 = nn.Conv1d(
            in_channels, filter_sizes[0], kernel_size=kernel_sizes[0]
        )
        self.bn1 = nn.BatchNorm1d(filter_sizes[0])
        self.drop1 = nn.Dropout(self.dropout["conv"][0])
        self.dense1 = nn.Linear(filter_sizes[0], in_channels)

        # arm 2: SimpleRNN context
        self.rnn = nn.RNN(
            in_channels, rnn_layer, batch_first=True, nonlinearity=self.rnn_nonlinearity
        )
        self.bn_rnn = nn.BatchNorm1d(rnn_layer)
        self.drop_rnn = nn.Dropout(self.dropout["rnn"])

        # projects rnn hidden state (rnn_layer) back to in_channels for concat
        self.dense_rnn = nn.Linear(rnn_layer, in_channels)

        # additional CCNN conv blocks applied after the arm1 + arm2 concat.
        self.conv_layers = nn.ModuleList()
        self.conv_dense_layers = nn.ModuleList()
        self.conv_bn_layers = nn.ModuleList()
        self.conv_drop_layers = nn.ModuleList()
        for kernel_size, filter_size, dropout_rate in zip(
            kernel_sizes[1:], filter_sizes[1:], self.dropout["conv"][1:]
        ):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filter_size, kernel_size=kernel_size)
            )
            self.conv_dense_layers.append(nn.Linear(filter_size, in_channels))
            self.conv_bn_layers.append(nn.BatchNorm1d(in_channels))
            self.conv_drop_layers.append(nn.Dropout(dropout_rate))

        # arm 3: CLSTM
        self.lstm = nn.LSTM(in_channels, lstm_layer, batch_first=True)

        # projects lstm hidden state (lstm_layer) back to in_channels for concat
        self.dense_lstm = nn.Linear(lstm_layer, in_channels)
        self.drop3 = nn.Dropout(self.dropout["lstm"])

        # post-merge
        self.pool = nn.MaxPool1d(
            kernel_size=avg_pool_size, stride=1, padding=avg_pool_size // 2
        )
        self.drop4 = nn.Dropout(self.dropout["pool"])

        # self-attention operates on in_channels (the channel dim after all projections)
        self.att = SeqSelfAttention(
            embed_dim=in_channels,
            attention_width=10,
            activation_attention=activation_attention,
        )
        self.drop5 = nn.Dropout(self.dropout["attention"])

        # MLP head
        self.mlp1 = nn.LazyLinear(dense_size)
        self.drop6 = nn.Dropout(self.dropout["mlp"])
        self.mlp2 = nn.Linear(dense_size, dense_size)
        self.drop7 = nn.Dropout(self.dropout["mlp"])
        self.out = nn.Linear(dense_size, n_classes)

        if init_weights is not None:
            self.apply(self._init_weights)

    def forward(self, x1, x3):
        """Forward pass.

        Parameters
        ----------
        x1 : torch.Tensor, shape [B, T, in_channels]
            Raw standardised time series input (fed to arms 1 and 2).
        x3 : torch.Tensor, shape [B, T, in_channels]
            Rolling-mean-augmented input (fed to arm 3).

        Returns
        -------
        torch.Tensor, shape [B, n_classes]
            Raw class logits (apply softmax externally or use CrossEntropyLoss).
        """
        B = x1.size(0)

        # arm 1: Conv1D block
        c1 = self._act(self.conv1(x1.permute(0, 2, 1)))
        c1 = self.bn1(c1)
        c1 = self.drop1(c1)
        c1 = self.dense1(c1.permute(0, 2, 1))

        # arm 2: SimpleRNN context
        rnn_out, _ = self.rnn(x1)
        rnn_out = rnn_out[:, -1, :]
        rnn_out = self.bn_rnn(rnn_out)
        rnn_out = self.drop_rnn(rnn_out)
        rnn_out = self.dense_rnn(rnn_out)
        rnn_out = rnn_out.unsqueeze(1)

        # CCNN merge: arm1 + arm2 on time axis
        conc1 = torch.cat([c1, rnn_out], dim=1)

        # Additional post-merge conv blocks
        c2 = conc1
        for conv, dense, bn, drop in zip(
            self.conv_layers,
            self.conv_dense_layers,
            self.conv_bn_layers,
            self.conv_drop_layers,
        ):
            c2 = self._act(conv(c2.permute(0, 2, 1)))
            c2 = dense(c2.permute(0, 2, 1))
            c2 = bn(c2.permute(0, 2, 1)).permute(0, 2, 1)
            c2 = drop(c2)

        # arm 3: CLSTM
        lstm_out, _ = self.lstm(x3)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dense_lstm(lstm_out)
        lstm_out = lstm_out.unsqueeze(1)
        lstm_out = self.drop3(lstm_out)

        # merge all arms on time axis
        merge = torch.cat([c2, lstm_out], dim=1)
        merge = self.pool(merge.permute(0, 2, 1)).permute(0, 2, 1)
        merge = self.drop4(merge)

        # self-attention
        att = self.att(merge)
        att = self.drop5(att)

        # MLP head
        flat = att.reshape(B, -1)
        h = self._act(self.mlp1(flat))
        h = self.drop6(h)
        h = self._act(self.mlp2(h))
        h = self.drop7(h)
        return self.out(h)  # raw logits

    def _init_weights(self, module):
        """Initialize the weights of the network."""
        if self.init_weights is None:
            return

        init_fn = getattr(torch.nn.init, self.init_weights)

        if isinstance(module, (nn.RNN, nn.LSTM)):
            for name, param in module.named_parameters():
                if "weight" in name:
                    # Skip initialization for uninitialized parameters
                    if isinstance(param, nn.parameter.UninitializedParameter):
                        continue
                    init_fn(param)
                elif "bias" in name:
                    # Skip initialization for uninitialized parameters
                    if isinstance(param, nn.parameter.UninitializedParameter):
                        continue
                    nn.init.zeros_(param)
        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
            # Skip initialization for uninitialized parameters
            if isinstance(module.weight, nn.parameter.UninitializedParameter):
                return

            init_fn(module.weight)
            if module.bias is not None:
                if isinstance(module.bias, nn.parameter.UninitializedParameter):
                    return
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            # Handle standalone parameters like in SelfAttention
            if isinstance(module, nn.parameter.UninitializedParameter):
                return
            init_fn(module)
