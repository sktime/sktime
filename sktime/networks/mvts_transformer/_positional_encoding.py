"""Positional encodings for the multivariate time series transformer.

The encodings themselves are shared with other sktime networks and live in
``sktime.libs._torch_positional_encoding``. This module binds them to the
names, defaults and sequence-first layout used by the transformer of [1]_.

References
----------
.. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
   and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
   Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD
   Conference on Knowledge Discovery & Data Mining (KDD '21), 2114-2124.
   https://doi.org/10.1145/3447548.3467401
"""

__authors__ = ["gzerveas", "geetu040"]
__all__ = [
    "FixedPositionalEncoding",
    "LearnablePositionalEncoding",
    "get_pos_encoder",
]

from sktime.libs._torch_positional_encoding import (
    LearnablePositionalEncoding as _LearnablePositionalEncoding,
)
from sktime.libs._torch_positional_encoding import (
    SinusoidalPositionalEncoding,
)


class FixedPositionalEncoding(SinusoidalPositionalEncoding):
    """Fixed sinusoidal positional encoding, in sequence-first layout.

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
    Input and output tensors have shape ``(seq_len, batch_size, d_model)``.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
            scale_factor=scale_factor,
            freq_scaling=1.0,
            batch_first=False,
        )


class LearnablePositionalEncoding(_LearnablePositionalEncoding):
    """Learnable positional encoding, in sequence-first layout.

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
    ``(seq_len, batch_size, d_model)``.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__(
            d_model=d_model, dropout=dropout, max_len=max_len, batch_first=False
        )


def get_pos_encoder(pos_encoding):
    """Look up a positional encoding class by name.

    Parameters
    ----------
    pos_encoding : str
        Name of the positional encoding, ``"fixed"`` or ``"learnable"``.

    Returns
    -------
    class
        The positional encoding class, not instantiated.

    Raises
    ------
    NotImplementedError
        If ``pos_encoding`` is not one of the supported names.
    """
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        f"pos_encoding should be 'learnable'/'fixed', not '{pos_encoding}'"
    )
