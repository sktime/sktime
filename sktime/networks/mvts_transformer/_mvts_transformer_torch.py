"""Multivariate time series transformer network in PyTorch, as described in [1]_.

Wrapped around the official PyTorch implementation of the transformer from [2]_,
provided by the authors of the paper [1]_.

References
----------
.. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
   and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
   Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD
   Conference on Knowledge Discovery & Data Mining (KDD '21). Association for
   Computing Machinery, New York, NY, USA, 2114-2124.
   https://doi.org/10.1145/3447548.3467401
.. [2] https://github.com/gzerveas/mvts_transformer
"""

__authors__ = ["gzerveas", "geetu040"]
__all__ = ["MVTSTransformerNetworkTorch", "TSTransformerEncoderTorch"]

import math
from collections.abc import Callable

from sktime.networks.mvts_transformer._encoder_layer import (
    TransformerBatchNormEncoderLayer,
)
from sktime.networks.mvts_transformer._positional_encoding import get_pos_encoder
from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class _TSTransformerEncoderBase(NNModule):
    """Shared transformer trunk of the multivariate time series transformer.

    Projects the input series into ``d_model`` dimensions, adds a positional
    encoding, and runs a stack of transformer encoder layers. Subclasses add
    the task specific output head.

    Parameters
    ----------
    feat_dim : int
        Number of input channels, i.e., variables of the time series.
    max_len : int
        Length of the input series.
    d_model : int
        Number of expected features in the transformer, i.e., the dimension
        the input is projected to.
    n_heads : int
        Number of heads in the multi-head attention.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Dimension of the feed-forward network of the encoder layers.
    dropout : float, default=0.1
        Dropout rate applied in the encoder layers and before the output head.
    pos_encoding : str, default="fixed"
        Type of positional encoding, ``"fixed"`` or ``"learnable"``.
    activation : Callable or None, default=None
        Activation applied to the output layer. If None, the network returns
        raw outputs, i.e., logits.
    activation_hidden : Callable or None, default=None
        Activation applied in the encoder layers and after the encoder stack.
        If None, no activation is applied.
    norm : str, default="BatchNorm"
        Normalization used in the encoder layers, ``"BatchNorm"`` or
        ``"LayerNorm"``.
    freeze : bool, default=False
        If True, dropout is disabled in the positional encoding and the
        encoder layers, as in the reference implementation.
    """

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super().__init__()

        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnIdentity = _safe_import("torch.nn.Identity")
        nnTransformerEncoder = _safe_import("torch.nn.TransformerEncoder")
        nnTransformerEncoderLayer = _safe_import("torch.nn.TransformerEncoderLayer")

        self.feat_dim = feat_dim
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.activation = activation

        self.project_inp = nnLinear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        act_hidden = nnIdentity() if activation_hidden is None else activation_hidden

        if norm == "LayerNorm":
            encoder_layer = nnTransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=act_hidden,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation_hidden=act_hidden,
            )

        self.transformer_encoder = nnTransformerEncoder(encoder_layer, num_layers)

        self.act = act_hidden

        self.dropout1 = nnDropout(dropout)

    def _encode(self, X, padding_masks):
        """Project, position-encode and transform the input series.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, feat_dim)
            Input tensor containing the time series data.
        padding_masks : torch.Tensor of shape (batch_size, seq_len)
            Boolean mask, True at non-padding positions.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, d_model)
            Encoder output.
        """
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)

        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        return output

    def _apply_output_activation(self, output):
        """Apply the output layer activation, if one was passed."""
        if self.activation is not None:
            output = self.activation(output)
        return output


class TSTransformerEncoderTorch(_TSTransformerEncoderBase):
    """Multivariate time series transformer with a reconstruction head.

    Projects the encoder output back to the input dimension, as used for the
    unsupervised pre-training objective of [1]_.

    Parameters
    ----------
    feat_dim : int
        Number of input channels, i.e., variables of the time series.
    max_len : int
        Length of the input series.
    d_model : int
        Number of expected features in the transformer.
    n_heads : int
        Number of heads in the multi-head attention.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Dimension of the feed-forward network of the encoder layers.
    dropout : float, default=0.1
        Dropout rate applied in the encoder layers and before the output head.
    pos_encoding : str, default="fixed"
        Type of positional encoding, ``"fixed"`` or ``"learnable"``.
    activation : Callable or None, default=None
        Activation applied to the output layer.
    activation_hidden : Callable or None, default=None
        Activation applied in the encoder layers and after the encoder stack.
    norm : str, default="BatchNorm"
        Normalization used in the encoder layers.
    freeze : bool, default=False
        If True, dropout is disabled in the positional encoding and the
        encoder layers.

    References
    ----------
    .. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
       and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
       Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD
       Conference on Knowledge Discovery & Data Mining (KDD '21), 2114-2124.
       https://doi.org/10.1145/3447548.3467401
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nnLinear = _safe_import("torch.nn.Linear")
        self.output_layer = nnLinear(self.d_model, self.feat_dim)

    def forward(self, X, padding_masks):
        """Reconstruct the input series.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, feat_dim)
            Input tensor containing the time series data.
        padding_masks : torch.Tensor of shape (batch_size, seq_len)
            Boolean mask, True at non-padding positions.

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len, feat_dim)
            Reconstruction of the input series.
        """
        output = self._encode(X, padding_masks)
        output = self.output_layer(output)
        return self._apply_output_activation(output)


class MVTSTransformerNetworkTorch(_TSTransformerEncoderBase):
    """Multivariate time series transformer with a classification head.

    Flattens the encoder output over the time axis and projects it to
    ``num_classes`` outputs, as described in [1]_.

    Parameters
    ----------
    feat_dim : int
        Number of input channels, i.e., variables of the time series.
    max_len : int
        Length of the input series.
    num_classes : int
        Number of outputs, i.e., the number of classes. Use ``1`` for regression.
    d_model : int, default=256
        Number of expected features in the transformer.
    n_heads : int, default=4
        Number of heads in the multi-head attention.
    num_layers : int, default=4
        Number of transformer encoder layers.
    dim_feedforward : int, default=128
        Dimension of the feed-forward network of the encoder layers.
    dropout : float, default=0.1
        Dropout rate applied in the encoder layers and before the output head.
    pos_encoding : str, default="fixed"
        Type of positional encoding, ``"fixed"`` or ``"learnable"``.
    activation : Callable or None, default=None
        Activation applied to the output layer. If None, the network returns
        raw outputs, i.e., logits.
    activation_hidden : Callable or None, default=None
        Activation applied in the encoder layers and after the encoder stack.
    norm : str, default="BatchNorm"
        Normalization used in the encoder layers.
    freeze : bool, default=False
        If True, dropout is disabled in the positional encoding and the
        encoder layers.

    References
    ----------
    .. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
       and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
       Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD
       Conference on Knowledge Discovery & Data Mining (KDD '21), 2114-2124.
       https://doi.org/10.1145/3447548.3467401
    """

    _tags = {
        "authors": ["gzerveas", "geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch"],
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: Callable | None = None,
        activation_hidden: Callable | None = None,
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super().__init__(
            feat_dim=feat_dim,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            activation=activation,
            activation_hidden=activation_hidden,
            norm=norm,
            freeze=freeze,
        )

        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        """Build the output head of the network.

        Parameters
        ----------
        d_model : int
            Number of expected features in the transformer.
        max_len : int
            Length of the input series.
        num_classes : int
            Number of outputs, i.e., the number of classes.

        Returns
        -------
        torch.nn.Linear
            The output layer, mapping the flattened encoder output to
            ``num_classes`` outputs.
        """
        nnLinear = _safe_import("torch.nn.Linear")
        return nnLinear(d_model * max_len, num_classes)

    def forward(self, X, padding_masks):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_len, feat_dim)
            Input tensor containing the time series data.
        padding_masks : torch.Tensor of shape (batch_size, seq_len)
            Boolean mask, True at non-padding positions.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Class scores. Raw outputs, i.e., logits, unless ``activation``
            is passed.
        """
        output = self._encode(X, padding_masks)

        output = output * padding_masks.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.output_layer(output)
        return self._apply_output_activation(output)
