"""Sequential self-attention layer implemented in PyTorch."""

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn_init = _safe_import("torch.nn.init")
NNModule = _safe_import("torch.nn.Module")
nn = _safe_import("torch.nn")


class SeqSelfAttentionTorch(NNModule):
    """Sequential self-attention layer.

    Parameters
    ----------
    units : int, default=32
        Dimension of the vectors used to calculate attention weights.
    attention_width : int or None, default=None
        Width of local attention.
    attention_type : str, default="additive"
        "additive" or "multiplicative".
    return_attention : bool, default=False
        Whether to return attention weights.
    history_only : bool, default=False
        Only use historical pieces of data.
    use_additive_bias : bool, default=True
        Whether to use bias in additive mode.
    use_attention_bias : bool, default=True
        Whether to use bias for attention weights.
    attention_activation : str or callable or torch.nn.Module or None, default=None
        Activation applied to attention logits.
    attention_regularizer_weight : float, default=0.0
        Weight for the attention regularizer.
    """

    ATTENTION_TYPE_ADD = "additive"
    ATTENTION_TYPE_MUL = "multiplicative"

    def __init__(
        self,
        units=32,
        attention_width=None,
        attention_type=ATTENTION_TYPE_ADD,
        return_attention=False,
        history_only=False,
        use_additive_bias=True,
        use_attention_bias=True,
        attention_activation=None,
        attention_regularizer_weight=0.0,
    ):
        """Layer initialization."""
        super().__init__()
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if self.history_only and self.attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.attention_regularizer_weight = attention_regularizer_weight
        self._attention_activation = self._instantiate_attention_activation(
            attention_activation
        )

        self._built = False
        self._input_dim = None

        # For inspection/debugging
        self.intensity = None
        self.attention = None
        self.attention_regularizer_loss = None

        if attention_type == SeqSelfAttentionTorch.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttentionTorch.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError(
                "No implementation for attention type : " + attention_type
            )

    def _instantiate_attention_activation(self, activation):
        """Instantiate the attention activation function."""
        if activation is None:
            return None
        if isinstance(activation, NNModule):
            return activation
        if callable(activation):
            return activation
        if isinstance(activation, str):
            act = activation.lower()
            if act == "tanh":
                return _safe_import("torch.nn.Tanh")()
            if act == "relu":
                return _safe_import("torch.nn.ReLU")()
            if act == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            if act in ("linear",):
                return None
        raise ValueError(
            "Unsupported attention_activation. Supported: "
            "'tanh', 'relu', 'sigmoid', 'linear', callable, or torch.nn.Module. "
            f"Found {activation}"
        )

    def _build(self, input_dim):
        """Build the layer parameters."""
        self._input_dim = int(input_dim)

        if self.attention_type == self.ATTENTION_TYPE_ADD:
            self.Wt = nn.Parameter(torch.empty(self._input_dim, self.units))
            self.Wx = nn.Parameter(torch.empty(self._input_dim, self.units))
            if self.use_additive_bias:
                self.bh = nn.Parameter(torch.empty(self.units))
            self.Wa = nn.Parameter(torch.empty(self.units, 1))
            if self.use_attention_bias:
                self.ba = nn.Parameter(torch.empty(1))
        elif self.attention_type == self.ATTENTION_TYPE_MUL:
            self.Wa = nn.Parameter(torch.empty(self._input_dim, self._input_dim))
            if self.use_attention_bias:
                self.ba = nn.Parameter(torch.empty(1))

        self._reset_parameters()
        self._built = True

    def _reset_parameters(self):
        """Initialize the layer parameters."""
        if self.Wt is not None:
            nn_init.xavier_normal_(self.Wt)
        if self.Wx is not None:
            nn_init.xavier_normal_(self.Wx)
        if self.Wa is not None:
            nn_init.xavier_normal_(self.Wa)
        if self.bh is not None:
            nn_init.zeros_(self.bh)
        if self.ba is not None:
            nn_init.zeros_(self.ba)

    def forward(self, x, mask=None):
        """Compute the output of the layer.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, time, features)
            Input sequence.
        mask : torch.Tensor or None
            Optional mask of shape (batch, time), where 1 indicates valid steps.
        """
        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, time, features). "
                f"Found {tuple(x.shape)}"
            )

        if not self._built:
            self._build(x.shape[-1])

        if self.attention_type == self.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(x)
        elif self.attention_type == self.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(x)

        if self._attention_activation is not None:
            e = self._attention_activation(e)

        if self.attention_width is not None:
            e = self._apply_attention_width(e, x.shape[1])

        if mask is not None:
            e = self._apply_mask(e, mask)

        self.intensity = e
        e = e - e.max(dim=-1, keepdim=True)[0]
        a = torch.exp(e)
        a = a / (a.sum(dim=-1, keepdim=True) + torch.finfo(a.dtype).eps)
        self.attention = a

        v = torch.bmm(a, x)
        if self.attention_regularizer_weight > 0.0:
            self.attention_regularizer_loss = self._attention_regularizer(a)
        else:
            self.attention_regularizer_loss = None

        if self.return_attention:
            return v, a
        return v

    def _call_additive_emission(self, x):
        # x: (B, T, F)
        q = torch.matmul(x, self.Wt)  # (B, T, U)
        k = torch.matmul(x, self.Wx)  # (B, T, U)
        q = q.unsqueeze(2)  # (B, T, 1, U)
        k = k.unsqueeze(1)  # (B, 1, T, U)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.bh)
        else:
            h = torch.tanh(q + k)
        if self.use_attention_bias:
            e = torch.matmul(h, self.Wa).squeeze(-1) + self.ba
        else:
            e = torch.matmul(h, self.Wa).squeeze(-1)
        return e

    def _call_multiplicative_emission(self, x):
        # e_{t, t'} = x_t^T W_a x_{t'}
        e = torch.matmul(x, self.Wa)  # (B, T, F)
        e = torch.matmul(e, x.transpose(1, 2))  # (B, T, T)
        if self.use_attention_bias:
            e = e + self.ba
        return e

    def _apply_attention_width(self, e, seq_len):
        """Apply attention width."""
        # e: (B, T, T)
        indices = torch.arange(seq_len, device=e.device)
        if self.history_only:
            lower = indices - (self.attention_width - 1)
        else:
            lower = indices - self.attention_width // 2
        lower = lower.unsqueeze(-1)  # (T, 1)
        upper = lower + self.attention_width
        idx = indices.unsqueeze(0)  # (1, T)
        local_mask = (lower <= idx) & (idx < upper)  # (T, T)
        e = e - 10000.0 * (~local_mask).to(e.dtype).unsqueeze(0)
        return e

    def _apply_mask(self, e, mask):
        """Apply mask to attention logits."""
        # mask: (B, T)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.to(dtype=e.dtype)
        mask = mask.unsqueeze(-1)
        e = e - 10000.0 * ((1.0 - mask) * (1.0 - mask.transpose(1, 2)))
        return e

    def _attention_regularizer(self, attention):
        """Orthogonality regularizer for attention weights."""
        batch_size = attention.shape[0]
        seq_len = attention.shape[-1]
        eye = torch.eye(seq_len, device=attention.device, dtype=attention.dtype)
        prod = torch.bmm(attention, attention.transpose(1, 2))
        reg = (prod - eye).pow(2).sum() / batch_size
        return self.attention_regularizer_weight * reg

    @staticmethod
    def get_custom_objects():
        """Return the custom objects of the layer."""
        return {"SeqSelfAttentionTorch": SeqSelfAttentionTorch}
