"""Attention components for ConvTran."""

__authors__ = ["srupat"]
__all__ = ["Attention", "AttentionRelScalar", "AttentionRelVector"]

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class _AttentionBase(NNModule):
    """Base class providing cached torch op imports for attention blocks."""

    def __init__(self):
        super().__init__()
        self._import_cache = {}

    def _torch_op(self, import_path):
        if import_path not in self._import_cache:
            self._import_cache[import_path] = _safe_import(import_path)
        return self._import_cache[import_path]


class Attention(_AttentionBase):
    """Multi-head self-attention block used in ConvTran.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float, default=0.0
        Dropout rate applied to attention weights.

    Notes
    -----
    Input and output tensors have shape ``(batch_size, seq_len, emb_size)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, emb_size, num_heads, dropout):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.scale = emb_size**-0.5
        super().__init__()

        nnLinear = _safe_import("torch.nn.Linear")
        self.key = nnLinear(emb_size, emb_size, bias=False)
        self.value = nnLinear(emb_size, emb_size, bias=False)
        self.query = nnLinear(emb_size, emb_size, bias=False)

        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(dropout)

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.to_out = nnLayerNorm(emb_size)

    def forward(self, x):
        """Compute attention over the sequence.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, emb_size)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, emb_size)
            Tensor after applying attention.
        """
        batch_size, seq_len, _ = x.shape
        # (batch, seq_len, emb_size) -> (batch, heads, head_dim, seq_len) for k
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        # (batch, seq_len, emb_size) -> (batch, heads, seq_len, head_dim) for q, v
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        torch_matmul = self._torch_op("torch.matmul")
        attn = torch_matmul(q, k) * self.scale
        softmax = self._torch_op("torch.nn.functional.softmax")
        attn = self.dropout(softmax(attn, dim=-1))

        out = torch_matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class AttentionRelScalar(_AttentionBase):
    """Attention with scalar relative position bias used in ConvTran.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    seq_len : int
        Sequence length used to construct the relative bias table.
    dropout : float, default=0.0
        Dropout rate applied to attention weights.

    Notes
    -----
    Input and output tensors have shape ``(batch_size, seq_len, emb_size)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.scale = emb_size**-0.5
        super().__init__()

        nnLinear = _safe_import("torch.nn.Linear")
        self.key = nnLinear(emb_size, emb_size, bias=False)
        self.value = nnLinear(emb_size, emb_size, bias=False)
        self.query = nnLinear(emb_size, emb_size, bias=False)

        torch_zeros = _safe_import("torch.zeros")
        nnParameter = _safe_import("torch.nn.Parameter")
        self.relative_bias_table = nnParameter(
            torch_zeros((2 * self.seq_len - 1), num_heads)
        )

        torch_arange = _safe_import("torch.arange")
        torch_meshgrid = _safe_import("torch.meshgrid")
        coords = torch_meshgrid(
            torch_arange(1),
            torch_arange(self.seq_len),
            indexing="ij",
        )
        torch_stack = _safe_import("torch.stack")
        coords = torch_stack(coords).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(dropout)

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.to_out = nnLayerNorm(emb_size)

    def forward(self, x):
        """Compute attention with scalar relative position bias.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, emb_size)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, emb_size)
            Tensor after applying attention.
        """
        batch_size, seq_len, _ = x.shape
        # (batch, seq_len, emb_size) -> (batch, heads, head_dim, seq_len) for k
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        # (batch, seq_len, emb_size) -> (batch, heads, seq_len, head_dim) for q, v
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        torch_matmul = self._torch_op("torch.matmul")
        attn = torch_matmul(q, k) * self.scale
        softmax = self._torch_op("torch.nn.functional.softmax")
        attn = softmax(attn, dim=-1)

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads)
        )
        relative_bias = (
            relative_bias.view(self.seq_len, self.seq_len, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attn = attn + relative_bias
        attn = self.dropout(attn)

        out = torch_matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class AttentionRelVector(_AttentionBase):
    """Attention with vector relative position encoding used in ConvTran.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    seq_len : int
        Sequence length used to construct relative position embeddings.
    dropout : float, default=0.0
        Dropout rate applied to attention weights.

    Notes
    -----
    Input and output tensors have shape ``(batch_size, seq_len, emb_size)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.scale = emb_size**-0.5
        super().__init__()

        nnLinear = _safe_import("torch.nn.Linear")
        self.key = nnLinear(emb_size, emb_size, bias=False)
        self.value = nnLinear(emb_size, emb_size, bias=False)
        self.query = nnLinear(emb_size, emb_size, bias=False)

        torch_randn = _safe_import("torch.randn")
        nnParameter = _safe_import("torch.nn.Parameter")
        head_dim = int(emb_size / num_heads)
        self.Er = nnParameter(torch_randn(self.seq_len, head_dim))

        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(dropout)

        nnLayerNorm = _safe_import("torch.nn.LayerNorm")
        self.to_out = nnLayerNorm(emb_size)

    def forward(self, x):
        """Compute attention with vector relative position encoding.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, emb_size)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, emb_size)
            Tensor after applying attention.
        """
        batch_size, seq_len, _ = x.shape
        # (batch, seq_len, emb_size) -> (batch, heads, head_dim, seq_len) for k
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        # (batch, seq_len, emb_size) -> (batch, heads, seq_len, head_dim) for q, v
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        torch_matmul = self._torch_op("torch.matmul")
        q_er = torch_matmul(q, self.Er.transpose(0, 1))
        s_rel = self._skew(q_er)
        attn = (torch_matmul(q, k) + s_rel) * self.scale
        softmax = self._torch_op("torch.nn.functional.softmax")
        attn = self.dropout(softmax(attn, dim=-1))

        out = torch_matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out

    def _skew(self, q_er):
        """Skew relative logits to align positions.

        Parameters
        ----------
        q_er : torch.Tensor of shape (batch_size, num_heads, seq_len, seq_len)
            Relative logits.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_heads, seq_len, seq_len)
            Skewed relative logits.
        """
        pad = self._torch_op("torch.nn.functional.pad")
        padded = pad(q_er, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        # (batch, heads, seq_len, seq_len + 1) -> (batch, heads, seq_len + 1, seq_len)
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        s_rel = reshaped[:, :, 1:, :]
        return s_rel
