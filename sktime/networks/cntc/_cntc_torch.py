"""CNTC for Classification and Regression in PyTorch."""

__authors__ = ["fnhirwa"]
__all__ = ["CNTCNetworkTorch"]


from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
torch = _safe_import("torch")
nn = _safe_import("torch.nn")
torchF = _safe_import("torch.nn.functional")
NNModule = _safe_import("torch.nn.Module")


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

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        score = self.v(torch.tanh(q.unsqueeze(2) + k.unsqueeze(1))).squeeze(-1)
        if self.attention_width is not None:
            T = x.size(1)
            mask = torch.tril(
                torch.ones(T, T, device=x.device), self.attention_width - 1
            )
            mask = torch.triu(mask, -(self.attention_width - 1))
            score = score.masked_fill(mask == 0, float("-inf"))
        attn = self._act(score)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        return torch.bmm(attn, x)


class CNTCNetworkTorch(NNModule):
    """CNTC (Contextual Neural Networks for Time Series Classification).

    CNTC is a deep learning network architecture for time series classification.
    It combines two parallel arms — a Contextual CNN (CCNN) and a Contextual LSTM
    (CLSTM) — fused via concatenation, followed by self-attention and an MLP head.

    Architecture summary:
      - Arm 1+2 (CCNN): Conv1D on raw input + SimpleRNN context, merged on the
        time axis, then a second Conv1D block.
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
        Kernel size for each Conv1D layer (first and second conv block).
    rnn_layer : int, default=64
        Hidden size of the SimpleRNN used in the CCNN arm.
    lstm_layer : int, default=8
        Hidden size of the LSTM used in the CLSTM arm.
    evg_pool_size : int, default=1
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
    dropout : float or tuple of 7 floats, default=(0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.8)
        Dropout rate(s) applied in the network. If a single float, the same
        rate is applied everywhere. If a tuple, values correspond to:
        (conv1_dropout, rnn_dropout, conv2_dropout, lstm_dropout,
         pool_dropout, attention_dropout, mlp_dropout)
        where mlp_dropout is shared across both MLP hidden layers.
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
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        kernel_sizes: tuple = (1, 1),
        rnn_layer: int = 64,
        lstm_layer: int = 8,
        evg_pool_size: int = 1,
        n_conv_layers: int = 2,
        filter_sizes: tuple = (16, 8),
        dense_size: int = 64,
        activation: str = "relu",
        activation_attention: str = "sigmoid",
        dropout: float | tuple = (0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.8),
        init_weights: str | None = "xavier_uniform",
        random_state: int = 0,
    ):
        super().__init__()

        assert len(kernel_sizes) == n_conv_layers, (
            f"len(kernel_sizes)={len(kernel_sizes)}"
            f" must equal n_conv_layers={n_conv_layers}"
        )
        assert len(filter_sizes) == n_conv_layers, (
            f"len(filter_sizes)={len(filter_sizes)}"
            f" must equal n_conv_layers={n_conv_layers}"
        )

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.activation = activation
        self.activation_attention = activation_attention
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.evg_pool_size = evg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.dense_size = dense_size
        self.random_state = random_state
        self.rnn_layer = rnn_layer
        self.lstm_layer = lstm_layer
        self.init_weights = init_weights

        if isinstance(dropout, float):
            d = [dropout] * 7
        else:
            assert len(dropout) == 7, "dropout tuple must have exactly 7 values"
            d = list(dropout)
        (d_conv1, d_rnn, d_conv2, d_lstm, d_pool, d_att, d_mlp) = d

        self._act = getattr(torchF, activation)

        torch.manual_seed(random_state)

        # arm 1: first conv block
        self.conv1 = nn.Conv1d(
            in_channels, filter_sizes[0], kernel_size=kernel_sizes[0]
        )
        self.bn1 = nn.BatchNorm1d(filter_sizes[0])
        self.drop1 = nn.Dropout(d_conv1)
        self.dense1 = nn.Linear(filter_sizes[0], in_channels)

        # arm 2: SimpleRNN context
        self.rnn = nn.RNN(in_channels, rnn_layer, batch_first=True, nonlinearity="relu")
        self.bn_rnn = nn.BatchNorm1d(rnn_layer)
        self.drop_rnn = nn.Dropout(d_rnn)

        # projects rnn hidden state (rnn_layer) back to in_channels for concat
        self.dense_rnn = nn.Linear(rnn_layer, in_channels)

        # CCNN second conv block (applied after arm1 + arm2 concat on time axis)
        self.conv2 = nn.Conv1d(
            in_channels, filter_sizes[1], kernel_size=kernel_sizes[1]
        )
        self.dense2 = nn.Linear(filter_sizes[1], in_channels)

        # LazyBatchNorm1d: input shape not known until first forward pass
        self.bn2 = nn.LazyBatchNorm1d()
        self.drop2 = nn.Dropout(d_conv2)

        # arm 3: CLSTM
        self.lstm = nn.LSTM(in_channels, lstm_layer, batch_first=True)

        # projects lstm hidden state (lstm_layer) back to in_channels for concat
        self.dense_lstm = nn.Linear(lstm_layer, in_channels)
        self.drop3 = nn.Dropout(d_lstm)

        # post-merge
        self.pool = nn.MaxPool1d(
            kernel_size=evg_pool_size, stride=1, padding=evg_pool_size // 2
        )
        self.drop4 = nn.Dropout(d_pool)

        # self-attention operates on in_channels (the channel dim after all projections)
        self.att = SeqSelfAttention(
            embed_dim=in_channels, attention_width=10, activation=activation_attention
        )
        self.drop5 = nn.Dropout(d_att)

        # MLP head
        self.mlp1 = nn.LazyLinear(dense_size)
        self.drop6 = nn.Dropout(d_mlp)
        self.mlp2 = nn.Linear(dense_size, dense_size)
        self.drop7 = nn.Dropout(d_mlp)
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

        # second conv block
        c2 = self._act(self.conv2(conc1.permute(0, 2, 1)))
        c2 = self.dense2(c2.permute(0, 2, 1))
        c2 = self.bn2(c2.permute(0, 2, 1)).permute(0, 2, 1)
        c2 = self.drop2(c2)

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
        """Initialize weights for the network modules."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init_fn = getattr(torch.nn.init, self.init_weights)
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    init_fn = getattr(torch.nn.init, self.init_weights)
                    init_fn(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        if isinstance(module, SeqSelfAttention):
            for name, param in module.named_parameters():
                if "W_q" in name or "W_k" in name or "v" in name:
                    init_fn = getattr(torch.nn.init, self.init_weights)
                    init_fn(param.weight)
                    if param.bias is not None:
                        nn.init.zeros_(param.bias)
