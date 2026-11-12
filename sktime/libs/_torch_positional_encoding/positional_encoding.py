"""Positional encoding layers implemented in PyTorch.

These layers are shared between the sktime networks that need to inject
position information into a sequence embedding, currently ConvTran
(``sktime.networks.convtran``) and the multivariate time series transformer
(``sktime.networks.mvts_transformer``).
"""

__authors__ = ["gzerveas", "Navid-Foumani", "srupat"]
__all__ = [
    "LearnablePositionalEncoding",
    "SinusoidalPositionalEncoding",
]

import math

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class _BasePositionalEncoding(NNModule):
    """Base class for positional encodings, handling layout and dropout.

    Parameters
    ----------
    dropout : float
        Dropout rate applied after adding the positional encodings.
    batch_first : bool
        If True, inputs are ``(batch_size, seq_len, d_model)`` and the sequence
        axis is 1. If False, inputs are ``(seq_len, batch_size, d_model)`` and
        the sequence axis is 0.
    """

    def __init__(self, dropout, batch_first):
        super().__init__()
        nnDropout = _safe_import("torch.nn.Dropout")
        self.dropout = nnDropout(p=dropout)
        self.batch_first = batch_first

    @property
    def _seq_axis(self):
        """Axis of the input tensor that carries the sequence length."""
        return 1 if self.batch_first else 0


class SinusoidalPositionalEncoding(_BasePositionalEncoding):
    """Fixed sinusoidal positional encoding.

    Implements the sinusoidal encoding of [1]_. Setting ``freq_scaling`` to
    ``d_model / max_len`` yields the time-aware variant (tAPE) of [2]_, which
    keeps the embedding distance-aware for short series.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding the positional encodings.
    max_len : int, default=1024
        Maximum sequence length for which encodings are pre-computed.
    scale_factor : float, default=1.0
        Scaling factor applied to the pre-computed encodings.
    freq_scaling : float, default=1.0
        Scaling factor applied to the sinusoid frequencies. ``1.0`` gives the
        standard encoding, ``d_model / max_len`` gives tAPE.
    batch_first : bool, default=True
        If True, inputs are ``(batch_size, seq_len, d_model)``.
        If False, inputs are ``(seq_len, batch_size, d_model)``.

    References
    ----------
    .. [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
       Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you
       need. Advances in Neural Information Processing Systems, 30, 2017.
    .. [2] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=1024,
        scale_factor=1.0,
        freq_scaling=1.0,
        batch_first=True,
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.scale_factor = scale_factor
        self.freq_scaling = freq_scaling
        super().__init__(dropout=dropout, batch_first=batch_first)

        torch_zeros = _safe_import("torch.zeros")
        torch_arange = _safe_import("torch.arange")
        torch_float = _safe_import("torch.float")
        torch_exp = _safe_import("torch.exp")
        torch_sin = _safe_import("torch.sin")
        torch_cos = _safe_import("torch.cos")

        pe = torch_zeros(max_len, d_model)
        position = torch_arange(0, max_len, dtype=torch_float).unsqueeze(1)
        div_term = torch_exp(
            torch_arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch_sin((position * div_term) * freq_scaling)
        pe[:, 1::2] = torch_cos((position * div_term) * freq_scaling)

        # the buffer is stored in the layout of the inputs, so that it can be
        # added without a transpose on every forward pass
        if batch_first:
            pe = scale_factor * pe.unsqueeze(0)
        else:
            pe = scale_factor * pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, d_model)`` if
            ``batch_first``, else ``(seq_len, batch_size, d_model)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with positional encodings added.
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(_BasePositionalEncoding):
    """Learnable positional encoding.

    The encoding is a free parameter of shape ``(max_len, d_model)``, initialized
    uniformly on ``[-0.02, 0.02]``, and broadcast over the batch axis.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding the positional encodings.
    max_len : int, default=1024
        Maximum sequence length the encoding is defined for.
    batch_first : bool, default=True
        If True, inputs are ``(batch_size, seq_len, d_model)``.
        If False, inputs are ``(seq_len, batch_size, d_model)``.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, batch_first=True):
        self.d_model = d_model
        self.max_len = max_len
        super().__init__(dropout=dropout, batch_first=batch_first)

        torch_empty = _safe_import("torch.empty")
        nnParameter = _safe_import("torch.nn.Parameter")
        uniform_ = _safe_import("torch.nn.init.uniform_")

        self.pe = nnParameter(torch_empty(max_len, d_model))
        uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, d_model)`` if
            ``batch_first``, else ``(seq_len, batch_size, d_model)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with positional encodings added.
        """
        seq_len = x.size(self._seq_axis)
        if seq_len > self.max_len:
            raise ValueError(
                "Input sequence length exceeds the maximum supported by "
                "LearnablePositionalEncoding. "
                f"Got seq_len={seq_len} and max_len={self.max_len}."
            )
        if self.batch_first:
            x = x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            x = x + self.pe[:seq_len, :].unsqueeze(1)
        return self.dropout(x)
