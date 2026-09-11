"""Positional encoding components for ConvTran (PyTorch)."""

__authors__ = ["srupat"]
__all__ = ["tAPE", "AbsolutePositionalEncoding", "LearnablePositionalEncoding"]

import math

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class _BasePositionalEncoding(NNModule):
    """Base class for ConvTran positional encodings."""

    def __init__(self, dropout):
        super().__init__()
        nn = _safe_import("torch.nn")
        self.dropout = nn.Dropout(p=dropout)


class tAPE(_BasePositionalEncoding):
    """Time-aware absolute positional encoding (tAPE) used in ConvTran.

    Implements the scaled sinusoidal positional encoding described in [1]_.
    The sinusoidal frequencies are scaled by ``d_model / max_len`` as in the
    original ConvTran implementation.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding positional encodings.
    max_len : int, default=1024
        Maximum sequence length.
    scale_factor : float, default=1.0
        Scaling factor applied to the positional encoding values.

    Notes
    -----
    Input and output tensors have shape ``(batch_size, seq_len, d_model)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        self.d_model = d_model
        self.max_len = max_len
        self.scale_factor = scale_factor
        super().__init__(dropout=dropout)

        torch_zeros = _safe_import("torch.zeros")
        pe = torch_zeros(max_len, d_model)

        torch_arange = _safe_import("torch.arange")
        torch_float = _safe_import("torch.float")
        position = torch_arange(0, max_len, dtype=torch_float).unsqueeze(1)

        torch_exp = _safe_import("torch.exp")
        div_term = torch_exp(
            torch_arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        torch_sin = _safe_import("torch.sin")
        torch_cos = _safe_import("torch.cos")
        scaling = d_model / max_len
        pe[:, 0::2] = torch_sin((position * div_term) * scaling)
        pe[:, 1::2] = torch_cos((position * div_term) * scaling)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, d_model)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, d_model)
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AbsolutePositionalEncoding(_BasePositionalEncoding):
    """Absolute sinusoidal positional encoding used in ConvTran.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding positional encodings.
    max_len : int, default=1024
        Maximum sequence length.
    scale_factor : float, default=1.0
        Scaling factor applied to the positional encoding values.

    Notes
    -----
    Input and output tensors have shape ``(batch_size, seq_len, d_model)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        self.d_model = d_model
        self.max_len = max_len
        self.scale_factor = scale_factor
        super().__init__(dropout=dropout)

        torch_zeros = _safe_import("torch.zeros")
        pe = torch_zeros(max_len, d_model)

        torch_arange = _safe_import("torch.arange")
        torch_float = _safe_import("torch.float")
        position = torch_arange(0, max_len, dtype=torch_float).unsqueeze(1)

        torch_exp = _safe_import("torch.exp")
        div_term = torch_exp(
            torch_arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        torch_sin = _safe_import("torch.sin")
        torch_cos = _safe_import("torch.cos")
        pe[:, 0::2] = torch_sin(position * div_term)
        pe[:, 1::2] = torch_cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, d_model)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, d_model)
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(_BasePositionalEncoding):
    """Learnable positional encoding used in ConvTran.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, default=0.1
        Dropout rate applied after adding positional encodings.
    max_len : int, default=1024
        Maximum sequence length.

    Notes
    -----
    The learnable positional encoding parameter ``pe`` has shape
    ``(max_len, d_model)`` and is added to inputs of shape
    ``(batch_size, seq_len, d_model)``.

    References
    ----------
    .. [1] Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, and
       Mahsa Salehi. Improving position encoding of transformers for multivariate
       time series classification. Data Mining and Knowledge Discovery,
       38(1):22-48, 2024. https://doi.org/10.1007/s10618-023-00948-2
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        self.d_model = d_model
        self.max_len = max_len
        super().__init__(dropout=dropout)

        torch_empty = _safe_import("torch.empty")
        nnParameter = _safe_import("torch.nn.Parameter")
        uniform_ = _safe_import("torch.nn.init.uniform_")

        self.pe = nnParameter(torch_empty(max_len, d_model))
        uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, d_model)
            Input tensor containing embedded time series.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, d_model)
            Tensor with positional encodings added.
        """
        if x.size(1) > self.max_len:
            raise ValueError(
                "Input sequence length exceeds the maximum supported by "
                "LearnablePositionalEncoding. "
                f"Got seq_len={x.size(1)} and max_len={self.max_len}."
            )
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return self.dropout(x)
