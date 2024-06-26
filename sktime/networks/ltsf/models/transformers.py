"""Deep Learning Forecaster using LTSF-Transformer Model."""
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

        pass


class LTSFTransformerNetwork:
    """LTSF-Transformer Forecaster."""

    def __init__(self, configs):
        self.configs = configs

    def _build(self):
        return self._LTSFTransformerNetwork(self.configs)

    class _LTSFTransformerNetwork(nn_module):
        def __init__(self, configs):

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

            super().__init__()
            self.pred_len = configs.pred_len
            self.seq_len = configs.seq_len
            self.context_len = configs.context_len
            self.output_attention = configs.output_attention

            # Embedding
            self.enc_embedding = LTSFDataEmbedding(
                configs.enc_in,
                configs.d_model,
                configs.freq,
                configs.dropout,
                configs.mark_vocab_sizes,
                configs.temporal_encoding_type,
                configs.position_encoding,
                configs.temporal_encoding,
            )._build()
            self.dec_embedding = LTSFDataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.freq,
                configs.dropout,
                configs.mark_vocab_sizes,
                configs.temporal_encoding_type,
                configs.position_encoding,
                configs.temporal_encoding,
            )._build()

            # LTSFTransformerEncoder
            self.encoder = LTSFTransformerEncoder(
                [
                    LTSFTransformerEncoderLayer(
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention,
                            )._build(),
                            configs.d_model,
                            configs.n_heads,
                        )._build(),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )._build()
                    for _ in range(configs.e_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
            )._build()
            # LTSFTransformerDecoder
            self.decoder = LTSFTransformerDecoder(
                [
                    LTSFTransformerDecoderLayer(
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            )._build(),
                            configs.d_model,
                            configs.n_heads,
                        )._build(),
                        LTSFAttentionLayer(
                            LTSFFullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            )._build(),
                            configs.d_model,
                            configs.n_heads,
                        )._build(),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )._build()
                    for _ in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
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
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]