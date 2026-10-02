"""Transformer encoder layer with batch normalization, in PyTorch.

Encoder layer of the multivariate time series transformer of [1]_, which
replaces the layer normalization of the standard transformer encoder layer
with batch normalization over the feature axis.

References
----------
.. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
   and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
   Time Series Representation Learning. In Proceedings of the 27th ACM SIGKDD
   Conference on Knowledge Discovery & Data Mining (KDD '21), 2114-2124.
   https://doi.org/10.1145/3447548.3467401
"""

__authors__ = ["gzerveas", "geetu040"]
__all__ = ["TransformerBatchNormEncoderLayer"]

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class TransformerBatchNormEncoderLayer(NNModule):
    """Transformer encoder layer using batch normalization instead of layer norm.

    Parameters
    ----------
    d_model : int
        Number of expected features in the input.
    nhead : int
        Number of heads in the multi-head attention.
    dim_feedforward : int, default=2048
        Dimension of the feed-forward network.
    dropout : float, default=0.1
        Dropout rate applied in the attention and feed-forward blocks.
    activation_hidden : torch.nn.Module or None, default=None
        Activation applied in the feed-forward block. If None, no activation
        is applied.

    Notes
    -----
    Input and output tensors have shape ``(seq_len, batch_size, d_model)``.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation_hidden=None,
    ):
        super().__init__()

        nnMultiheadAttention = _safe_import("torch.nn.MultiheadAttention")
        nnLinear = _safe_import("torch.nn.Linear")
        nnDropout = _safe_import("torch.nn.Dropout")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnIdentity = _safe_import("torch.nn.Identity")

        self.self_attn = nnMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nnLinear(d_model, dim_feedforward)
        self.dropout = nnDropout(dropout)
        self.linear2 = nnLinear(dim_feedforward, d_model)

        self.norm1 = nnBatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nnBatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nnDropout(dropout)
        self.dropout2 = nnDropout(dropout)

        self.activation = (
            nnIdentity() if activation_hidden is None else activation_hidden
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=None,
    ):
        """Run one encoder layer.

        Parameters
        ----------
        src : torch.Tensor of shape (seq_len, batch_size, d_model)
            Input sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            Additive mask for the source sequence.
        src_key_padding_mask : torch.Tensor, optional
            Mask marking which source keys are padding.
        is_causal : bool, optional
            Accepted for signature compatibility with
            ``torch.nn.TransformerEncoder``, and not used.

        Returns
        -------
        torch.Tensor of shape (seq_len, batch_size, d_model)
            Output of the encoder layer.
        """
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
