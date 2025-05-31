"""Deep Learning Forecaster using LTSF-Transformer Model."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class LTSFTransformerNetwork:
    """LTSF-Transformer Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) transformer forecaster,
    aka LTSF-Transformer, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
        Preferred to be twice the pred_len.
    pred_len : int
        Length of the prediction sequence.
    context_len : int, optional (default=2)
        Length of the label sequence.
        Preferred to be same as the pred_len.
    position_encoding : bool, optional (default=True)
        Whether to use positional encoding.
        Positional encoding helps the model understand the order of elements
        in the input sequence by adding unique positional information to each element.
    temporal_encoding : bool, optional (default=True)
        Whether to use temporal encoding.
        Works only with DatetimeIndex and PeriodIndex, disabled otherwise.
    temporal_encoding_type : str, optional (default="linear")
        Type of temporal encoding to use, relevant only if temporal_encoding is True.
        - "linear": Uses linear layer to encode temporal data.
        - "embed": Uses embeddings layer with learnable weights.
        - "fixed-embed": Uses embeddings layer with fixed sine-cosine values as weights.
    d_model : int, optional (default=512)
        Dimension of the model.
    n_heads : int, optional (default=8)
        Number of attention heads.
    d_ff : int, optional (default=2048)
        Dimension of the feedforward network model.
    e_layers : int, optional (default=3)
        Number of encoder layers.
    d_layers : int, optional (default=2)
        Number of decoder layers.
    factor : int, optional (default=5)
        Factor for the attention mechanism.
    dropout : float, optional (default=0.1)
        Dropout rate.
    activation : str, optional (default="relu")
        Activation function to use. Defaults to relu and otherwise gelu.
    freq : str, optional (default="h")
        Frequency of the input data, relevant only if temporal_encoding is True.

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFTransformerNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            context_len,
            pred_len,
            output_attention,
            mark_vocab_sizes,
            position_encoding,
            temporal_encoding,
            temporal_encoding_type,
            enc_in,
            dec_in,
            d_model,
            n_heads,
            d_ff,
            e_layers,
            d_layers,
            factor,
            dropout,
            activation,
            c_out,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.context_len = context_len
            self.pred_len = pred_len
            self.output_attention = output_attention
            self.mark_vocab_sizes = mark_vocab_sizes
            self.position_encoding = position_encoding
            self.temporal_encoding = temporal_encoding
            self.temporal_encoding_type = temporal_encoding_type
            self.enc_in = enc_in
            self.dec_in = dec_in
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_ff = d_ff
            self.e_layers = e_layers
            self.d_layers = d_layers
            self.factor = factor
            self.dropout = dropout
            self.activation = activation
            self.c_out = c_out

            from sktime.networks.ltsf.layers.attention import (
                LTSFAttentionLayer,
                LTSFFullAttention,
            )
            from sktime.networks.ltsf.layers.embed import LTSFDataEmbedding
            from sktime.networks.ltsf.layers.enc_dec import (
                LTSFTransformerDecoder,
                LTSFTransformerDecoderLayer,
                LTSFTransformerEncoder,
                LTSFTransformerEncoderLayer,
            )

            # Embedding
            self.enc_embedding = LTSFDataEmbedding(
                self.enc_in,
                self.d_model,
                self.dropout,
                self.mark_vocab_sizes,
                self.temporal_encoding_type,
                self.position_encoding,
                self.temporal_encoding,
            )._build()
            self.dec_embedding = LTSFDataEmbedding(
                self.dec_in,
                self.d_model,
                self.dropout,
                self.mark_vocab_sizes,
                self.temporal_encoding_type,
                self.position_encoding,
                self.temporal_encoding,
            )._build()

            # LTSFTransformerEncoder
            self.encoder = LTSFTransformerEncoder(
                [
                    LTSFTransformerEncoderLayer(
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                False,
                                self.factor,
                                attention_dropout=self.dropout,
                                output_attention=self.output_attention,
                            )._build(),
                            self.d_model,
                            self.n_heads,
                        )._build(),
                        self.d_model,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation,
                    )._build()
                    for _ in range(self.e_layers)
                ],
                norm_layer=nn.LayerNorm(self.d_model),
            )._build()
            # LTSFTransformerDecoder
            self.decoder = LTSFTransformerDecoder(
                [
                    LTSFTransformerDecoderLayer(
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                True,
                                self.factor,
                                attention_dropout=self.dropout,
                                output_attention=False,
                            )._build(),
                            self.d_model,
                            self.n_heads,
                        )._build(),
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                False,
                                self.factor,
                                attention_dropout=self.dropout,
                                output_attention=False,
                            )._build(),
                            self.d_model,
                            self.n_heads,
                        )._build(),
                        self.d_model,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation,
                    )._build()
                    for _ in range(self.d_layers)
                ],
                norm_layer=nn.LayerNorm(self.d_model),
                projection=nn.Linear(self.d_model, self.c_out, bias=True),
            )._build()

        def forward(self, x):
            """Forward pass for LSTF-Transformer Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            x_enc = x["x_enc"]
            x_mark_enc = x["x_mark_enc"]
            x_dec = x["x_dec"]
            x_mark_dec = x["x_mark_dec"]

            return self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        def _forward(
            self,
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
        ):
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(
                dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
            )

            if self.output_attention:
                return dec_out[:, -self.pred_len :, :], attns
            else:
                return dec_out[:, -self.pred_len :, :]  # [B, L, D]

    def __init__(
        self,
        seq_len,
        context_len,
        pred_len,
        output_attention,
        mark_vocab_sizes,
        position_encoding,
        temporal_encoding,
        temporal_encoding_type,
        enc_in,
        dec_in,
        d_model,
        n_heads,
        d_ff,
        e_layers,
        d_layers,
        factor,
        dropout,
        activation,
        c_out,
    ):
        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.mark_vocab_sizes = mark_vocab_sizes
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.temporal_encoding_type = temporal_encoding_type
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.c_out = c_out

    def _build(self):
        return self._LTSFTransformerNetwork(
            seq_len=self.seq_len,
            context_len=self.context_len,
            pred_len=self.pred_len,
            output_attention=self.output_attention,
            mark_vocab_sizes=self.mark_vocab_sizes,
            position_encoding=self.position_encoding,
            temporal_encoding=self.temporal_encoding,
            temporal_encoding_type=self.temporal_encoding_type,
            enc_in=self.enc_in,
            dec_in=self.dec_in,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            factor=self.factor,
            dropout=self.dropout,
            activation=self.activation,
            c_out=self.c_out,
        )
