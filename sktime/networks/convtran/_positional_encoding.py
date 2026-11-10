"""Positional encoding components for ConvTran (PyTorch)."""

__authors__ = ["srupat"]
__all__ = ["tAPE", "AbsolutePositionalEncoding", "LearnablePositionalEncoding"]

from sktime.libs._torch_positional_encoding import (
    LearnablePositionalEncoding as _LearnablePositionalEncoding,
)
from sktime.libs._torch_positional_encoding import (
    SinusoidalPositionalEncoding,
)


class tAPE(SinusoidalPositionalEncoding):
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
        super().__init__(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
            scale_factor=scale_factor,
            freq_scaling=d_model / max_len,
            batch_first=True,
        )


class AbsolutePositionalEncoding(SinusoidalPositionalEncoding):
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
        super().__init__(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
            scale_factor=scale_factor,
            freq_scaling=1.0,
            batch_first=True,
        )


class LearnablePositionalEncoding(_LearnablePositionalEncoding):
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
        super().__init__(
            d_model=d_model, dropout=dropout, max_len=max_len, batch_first=True
        )
