"""Sequential self-attention layer implemented in PyTorch."""

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class SeqSelfAttentionTorch(NNModule):
    """Sequential self-attention layer for temporal feature refinement.

    The layer consumes a batch of sequences with shape ``(batch, time,
    features)`` and computes pairwise attention scores between time steps.

    Two attention scoring mechanisms are supported:

    - ``"additive"`` computes Bahdanau-style scores using learned projections.
    - ``"multiplicative"`` computes bilinear scores directly in feature space.

    Optional local-attention and causal-attention modes can restrict which time
    steps contribute to each output position.

    Parameters
    ----------
    input_dim : int
        Feature dimension of the expected input. This parameter is **required** so
        that all learnable weights are materialized and registered during
        construction.
    units : int, default=32
        Hidden dimension used when ``attention_type="additive"``. Larger values
        increase the capacity of the attention scorer.
    attention_width : int or None, default=None
        Size of the local attention window. If ``None``, every time step may
        attend to every other time step. When set, attention is restricted to a
        fixed-width neighborhood around each position.
    attention_type : str, default="additive"
        Attention scoring function to use. Must be ``"additive"`` or
        ``"multiplicative"``.
    return_attention : bool, default=False
        If ``True``, :meth:`forward` returns a tuple containing the transformed
        sequence and the attention weights.
    history_only : bool, default=False
        If ``True``, each time step may only attend to itself and earlier time
        steps. When enabled without an explicit ``attention_width``, the layer
        behaves like global causal attention.
    use_additive_bias : bool, default=True
        Whether to include a bias term in the hidden projection used by
        additive attention.
    use_attention_bias : bool, default=True
        Whether to include a bias term in the final attention logits.
    attention_activation : str or callable or torch.nn.Module or None, default=None
        Optional non-linearity applied to the attention logits before masking
        and softmax normalization. String shortcuts ``"tanh"``, ``"relu"``,
        ``"sigmoid"``, and ``"linear"`` are supported.
    attention_regularizer_weight : float, default=0.0
        Weight of an orthogonality-inspired penalty applied to the attention
        matrix. Set to ``0.0`` to disable the regularizer.
    """

    ATTENTION_TYPE_ADD = "additive"
    ATTENTION_TYPE_MUL = "multiplicative"

    def __init__(
        self,
        input_dim,
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
        super().__init__()
        self._import_cache = {}
        self._input_dim = int(input_dim)
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

        # For inspection/debugging
        self.intensity = None
        self.attention = None
        self.attention_regularizer_loss = None

        self.Wx = None
        self.Wt = None
        self.bh = None
        self.Wa = None
        self.ba = None

        if attention_type == SeqSelfAttentionTorch.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttentionTorch.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError(
                "No implementation for attention type : " + attention_type
            )

        self._build(self._input_dim)

    def _torch_op(self, import_path):
        """Return a lazily imported PyTorch object.

        Parameters
        ----------
        import_path : str
            Fully qualified import path understood by :func:`_safe_import`.

        Returns
        -------
        object
            Imported function or class cached for reuse.
        """
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]

    def _instantiate_attention_activation(self, activation):
        """Normalize the configured attention activation.

        Parameters
        ----------
        activation : str, callable, torch.nn.Module, or None
            User-provided activation specification.

        Returns
        -------
        callable or torch.nn.Module or None
            Instantiated activation object, callable, or ``None`` when the
            attention logits should remain linear.

        Raises
        ------
        ValueError
            If ``activation`` is not one of the supported string aliases and is
            not callable.
        """
        Module = self._torch_op("torch.nn.modules.module.Module")
        if activation is None:
            return None
        if isinstance(activation, Module):
            return activation
        if callable(activation):
            return activation
        if isinstance(activation, str):
            act = activation.lower()
            if act == "tanh":
                return self._torch_op("torch.nn.Tanh")()
            if act == "relu":
                return self._torch_op("torch.nn.ReLU")()
            if act == "sigmoid":
                return self._torch_op("torch.nn.Sigmoid")()
            if act in ("linear",):
                return None
        raise ValueError(
            "Unsupported attention_activation. Supported: "
            "'tanh', 'relu', 'sigmoid', 'linear', callable, or torch.nn.Module. "
            f"Found {activation}"
        )

    def _build(self, input_dim):
        """Create learnable parameters for a given input feature dimension.

        Parameters
        ----------
        input_dim : int
            Number of features in the last dimension of the input tensor.

        Raises
        ------
        ValueError
            If the layer has already been built for a different input
            dimensionality.
        """
        if self._built:
            if self._input_dim != int(input_dim):
                raise ValueError(
                    "SeqSelfAttentionTorch already built with input_dim="
                    f"{self._input_dim}, got {input_dim}."
                )
            return
        self._input_dim = int(input_dim)

        Parameter = self._torch_op("torch.nn.Parameter")
        torch_empty = self._torch_op("torch.empty")

        if self.attention_type == self.ATTENTION_TYPE_ADD:
            self.Wt = Parameter(torch_empty(self._input_dim, self.units))
            self.Wx = Parameter(torch_empty(self._input_dim, self.units))
            if self.use_additive_bias:
                self.bh = Parameter(torch_empty(self.units))
            self.Wa = Parameter(torch_empty(self.units, 1))
            if self.use_attention_bias:
                self.ba = Parameter(torch_empty(1))
        elif self.attention_type == self.ATTENTION_TYPE_MUL:
            self.Wa = Parameter(torch_empty(self._input_dim, self._input_dim))
            if self.use_attention_bias:
                self.ba = Parameter(torch_empty(1))

        self._reset_parameters()
        self._built = True

    def _reset_parameters(self):
        """Initialize learnable parameters with standard PyTorch defaults.

        Weight matrices use Xavier normal initialization and bias vectors are
        initialized to zero.
        """
        xavier_normal_ = self._torch_op("torch.nn.init.xavier_normal_")
        zeros_ = self._torch_op("torch.nn.init.zeros_")
        if self.Wt is not None:
            xavier_normal_(self.Wt)
        if self.Wx is not None:
            xavier_normal_(self.Wx)
        if self.Wa is not None:
            xavier_normal_(self.Wa)
        if self.bh is not None:
            zeros_(self.bh)
        if self.ba is not None:
            zeros_(self.ba)

    def forward(self, x, mask=None):
        """Apply sequential self-attention to an input batch.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, time, features)
            Input batch of temporal feature vectors.
        mask : torch.Tensor or None, default=None
            Optional validity mask with shape ``(batch, time)`` or
            ``(batch, time, 1)``. Entries with value 1 mark valid time steps;
            entries with value 0 suppress attention involving padded positions.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Attention-weighted sequence with shape ``(batch, time, features)``.
            If ``return_attention=True``, the method additionally returns the
            normalized attention matrix with shape ``(batch, time, time)``.

        Raises
        ------
        ValueError
            If ``x`` is not three-dimensional or if its feature dimension does
            not match ``input_dim`` provided during construction.
        """
        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, time, features). "
                f"Found {tuple(x.shape)}"
            )

        if x.shape[-1] != self._input_dim:
            raise ValueError(
                "Input feature dimension does not match layer input_dim. "
                f"Expected {self._input_dim}, got {x.shape[-1]}."
            )

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
        torch_exp = self._torch_op("torch.exp")
        torch_finfo = self._torch_op("torch.finfo")
        a = torch_exp(e)
        a = a / (a.sum(dim=-1, keepdim=True) + torch_finfo(a.dtype).eps)
        self.attention = a

        torch_bmm = self._torch_op("torch.bmm")
        v = torch_bmm(a, x)
        if self.attention_regularizer_weight > 0.0:
            self.attention_regularizer_loss = self._attention_regularizer(a)
        else:
            self.attention_regularizer_loss = None

        if self.return_attention:
            return v, a
        return v

    def _call_additive_emission(self, x):
        """Compute additive attention logits for every pair of time steps.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, time, features)
            Input sequence batch.

        Returns
        -------
        torch.Tensor, shape (batch, time, time)
            Unnormalized pairwise attention logits.
        """
        # x: (B, T, F)
        torch_matmul = self._torch_op("torch.matmul")
        torch_tanh = self._torch_op("torch.tanh")
        q = torch_matmul(x, self.Wt)  # (B, T, U)
        k = torch_matmul(x, self.Wx)  # (B, T, U)

        # For broadcasting
        q = q.unsqueeze(2)  # (B, T, 1, U)
        k = k.unsqueeze(1)  # (B, 1, T, U)

        if self.use_additive_bias:
            h = torch_tanh(q + k + self.bh)
        else:
            h = torch_tanh(q + k)
        if self.use_attention_bias:
            e = torch_matmul(h, self.Wa).squeeze(-1) + self.ba
        else:
            e = torch_matmul(h, self.Wa).squeeze(-1)
        return e

    def _call_multiplicative_emission(self, x):
        """Compute multiplicative attention logits for a sequence batch.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, time, features)
            Input sequence batch.

        Returns
        -------
        torch.Tensor, shape (batch, time, time)
            Unnormalized bilinear attention logits where
            ``e[t, t'] = x_t^T W_a x_t'``.
        """
        # e_{t, t'} = x_t^T W_a x_{t'}
        torch_matmul = self._torch_op("torch.matmul")
        e = torch_matmul(x, self.Wa)  # (B, T, F)
        e = torch_matmul(e, x.transpose(1, 2))  # (B, T, T)
        if self.use_attention_bias:
            e = e + self.ba
        return e

    def _apply_attention_width(self, e, seq_len):
        """Restrict attention scores to a local temporal window.

        Parameters
        ----------
        e : torch.Tensor, shape (batch, time, time)
            Attention logits before softmax normalization.
        seq_len : int
            Number of time steps in the sequence.

        Returns
        -------
        torch.Tensor, shape (batch, time, time)
            Attention logits with out-of-window entries shifted by a large
            negative value so that their softmax contribution is negligible.
        """
        # e: (B, T, T)
        torch_arange = self._torch_op("torch.arange")
        indices = torch_arange(seq_len, device=e.device)
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
        """Mask attention logits corresponding to padded sequence positions.

        Parameters
        ----------
        e : torch.Tensor, shape (batch, time, time)
            Attention logits before softmax normalization.
        mask : torch.Tensor
            Validity mask with shape ``(batch, time)`` or ``(batch, time, 1)``.

        Returns
        -------
        torch.Tensor, shape (batch, time, time)
            Attention logits with padded positions strongly down-weighted.
        """
        # mask: (B, T)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.to(dtype=e.dtype)
        valid_pairs = mask.unsqueeze(-1) * mask.unsqueeze(1)
        e = e - 10000.0 * (1.0 - valid_pairs)
        return e

    def _attention_regularizer(self, attention):
        """Compute the optional regularization loss for attention weights.

        The penalty encourages attention matrices to be close to orthogonal by
        comparing ``attention @ attention.T`` to the identity matrix.

        Parameters
        ----------
        attention : torch.Tensor, shape (batch, time, time)
            Normalized attention matrix.

        Returns
        -------
        torch.Tensor
            Scalar regularization term scaled by
            ``attention_regularizer_weight``.
        """
        batch_size = attention.shape[0]
        seq_len = attention.shape[-1]
        torch_eye = self._torch_op("torch.eye")
        torch_bmm = self._torch_op("torch.bmm")
        eye = torch_eye(seq_len, device=attention.device, dtype=attention.dtype)
        prod = torch_bmm(attention, attention.transpose(1, 2))
        reg = (prod - eye).pow(2).sum() / batch_size
        return self.attention_regularizer_weight * reg

    @staticmethod
    def get_custom_objects():
        """Return the custom object mapping for serialization helpers.

        Returns
        -------
        dict[str, type]
            Mapping from the public layer name to its implementing class.
        """
        return {"SeqSelfAttentionTorch": SeqSelfAttentionTorch}
