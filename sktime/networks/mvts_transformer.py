"""Multivariate Time Series Transformer Network for Classification, described in [1]_.

This classifier has been wrapped around the official pytorch implementation of
Transformer from [2]_, provided by the authors of the paper [1]_.

References
----------
.. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
and Carsten Eickhoff. 2021. A Transformer-based Framework
for Multivariate Time Series Representation Learning.
In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery
& Data Mining (KDD '21). Association for Computing Machinery, New York, NY, USA,
2114-2124. https://doi.org/10.1145/3447548.3467401.
.. [2] https://github.com/gzerveas/mvts_transformer
"""

import math

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.nn.modules import (
        BatchNorm1d,
        Dropout,
        Linear,
        MultiheadAttention,
        TransformerEncoderLayer,
    )

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(f"activation should be relu/gelu, not {activation}")


class FixedPositionalEncoding(nn_module):
    """FixedPositionalEncoding."""

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward Pass."""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn_module):
    """LearnablePositionalEncoding."""

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """Forward Pass."""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    """get_pos_encoder."""
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        f"pos_encoding should be 'learnable'/'fixed', not '{pos_encoding}'"
    )


class TransformerBatchNormEncoderLayer(nn_module):
    """TransformerBatchNormEncoderLayer."""

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        """Setstate."""
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=None,
    ):
        """Forward Pass."""
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = src.permute(1, 2, 0)

        src = self.norm1(src)
        src = src.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)
        return src


class TSTransformerEncoder(nn_module):
    """TSTransformerEncoder."""

    def __init__(
        self,
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """Forward Pass."""
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)

        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)

        output = self.output_layer(output)

        return output


class TSTransformerEncoderClassiregressor(nn_module):
    """TSTransformerEncoderClassiregressor."""

    def __init__(
        self,
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        """."""
        output_layer = nn.Linear(d_model * max_len, num_classes)

        return output_layer

    def forward(self, X, padding_masks):
        """Forward Pass."""
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)

        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)

        output = output * padding_masks.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.output_layer(output)

        return output
