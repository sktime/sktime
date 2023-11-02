"""Deep Learning Forecasters using LTSF-Linear Models."""
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

        pass


class LTSFLinearNetwork:
    """LTSF-Linear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) linear forecaster,
    aka LTSF-Linear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            super().__init__()

            self.seq_len = seq_len
            self.pred_len = pred_len
            self.in_channels = in_channels
            self.individual = individual

            if self.individual:
                self.Linear = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-Linear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            if self.individual:
                output = zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(
                    x.device
                )
                for i in range(self.in_channels):
                    output[:, :, i] = self.Linear[i](x[:, :, i])
                x = output
            else:
                x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x  # [Batch, Output Length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )


class LTSFDLinearNetwork:
    """LTSF-DLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) decomposition linear
    forecaster, aka LTSF-DLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFDLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            from layers import SeriesDecomposer

            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len

            # Decompsition Kernel Size
            kernel_size = 25
            self.decompsition = SeriesDecomposer(kernel_size)._build()
            self.individual = individual
            self.in_channels = in_channels

            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-DLinear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            # x: [Batch, Input length, Channel]
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(
                0, 2, 1
            ), trend_init.permute(0, 2, 1)
            if self.individual:
                seasonal_output = zeros(
                    [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                    dtype=seasonal_init.dtype,
                ).to(seasonal_init.device)
                trend_output = zeros(
                    [trend_init.size(0), trend_init.size(1), self.pred_len],
                    dtype=trend_init.dtype,
                ).to(trend_init.device)
                for i in range(self.in_channels):
                    seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                        seasonal_init[:, i, :]
                    )
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            else:
                seasonal_output = self.Linear_Seasonal(seasonal_init)
                trend_output = self.Linear_Trend(trend_init)

            x = seasonal_output + trend_output
            return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFDLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )


class LTSFNLinearNetwork:
    """LTSF-NLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) normalization linear
    forecaster, aka LTSF-NLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFNLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len

            self.in_channels = in_channels
            self.individual = individual

            if self.individual:
                self.Linear = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-NLinear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            # x: [Batch, Input length, Channel]
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last
            if self.individual:
                output = zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(
                    x.device
                )
                for i in range(self.in_channels):
                    output[:, :, i] = self.Linear[i](x[:, :, i])
                x = output
            else:
                x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + seq_last
            return x  # [Batch, Output length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFNLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )


class LTSFAutoformerNetwork:
    """Network for LTSFAutoFormer.

    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity.
    """

    class _LTSFAutoformerNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            label_len,
            pred_len,
            output_attention,
            moving_avg,
            embed_type,
            enc_in,
            dec_in,
            d_model,
            embed,
            freq,
            dropout,
            factor,
            n_heads,
            e_layers,
            d_layers,
            c_out,
            activation,
        ):
            super().__init__()

            from embed import (
                DataEmbedding,
                DataEmbeddingWithoutPositional,
                DataEmbeddingWithoutPositionalTemporal,
                DataEmbeddingWithoutTemporal,
            )
            from layers import (
                AutoCorrelation,
                AutoCorrelationLayer,
                Decoder,
                DecoderLayer,
                Encoder,
                EncoderLayer,
                LTSFLayerNorm,
                SeriesDecomposer,
            )

            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            self.output_attention = output_attention
            self.moving_avg = moving_avg
            self.embed_type = embed_type
            self.enc_in = enc_in
            self.dec_in = dec_in
            self.d_model = d_model
            self.embed = embed
            self.freq = freq
            self.dropout = dropout
            self.factor = factor
            self.n_heads = n_heads
            self.e_layers = e_layers
            self.d_layers = d_layers
            self.c_out = c_out
            self.activation = activation

            # Decomp
            kernel_size = self.moving_avg
            self.decomp = SeriesDecomposer(kernel_size)._build()

            # Embedding
            # The series-wise connection inherently contains the sequential information.
            # Thus, we can discard the position embedding of transformers.
            if self.embed_type == 0:
                self.enc_embedding = DataEmbeddingWithoutPositional(
                    self.enc_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
                self.dec_embedding = DataEmbeddingWithoutPositional(
                    self.dec_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
            elif self.embed_type == 1:
                self.enc_embedding = DataEmbedding(
                    self.enc_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
                self.dec_embedding = DataEmbedding(
                    self.dec_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
            elif self.embed_type == 2:
                self.enc_embedding = DataEmbeddingWithoutPositional(
                    self.enc_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
                self.dec_embedding = DataEmbeddingWithoutPositional(
                    self.dec_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
            elif self.embed_type == 3:
                self.enc_embedding = DataEmbeddingWithoutTemporal(
                    self.enc_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
                self.dec_embedding = DataEmbeddingWithoutTemporal(
                    self.dec_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
            elif self.embed_type == 4:
                self.enc_embedding = DataEmbeddingWithoutPositionalTemporal(
                    self.enc_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()
                self.dec_embedding = DataEmbeddingWithoutPositionalTemporal(
                    self.dec_in, self.d_model, self.embed, self.freq, self.dropout
                )._build()

            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
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
                        moving_avg=self.moving_avg,
                        dropout=self.dropout,
                        activation=self.activation,
                    )._build()
                    for _ in range(self.e_layers)
                ],
                norm_layer=LTSFLayerNorm(self.d_model)._build(),
            )._build()
            # Decoder
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                True,
                                self.factor,
                                attention_dropout=self.dropout,
                                output_attention=False,
                            )._build(),
                            self.d_model,
                            self.n_heads,
                        )._build(),
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                False,
                                self.factor,
                                attention_dropout=self.dropout,
                                output_attention=False,
                            )._build(),
                            self.d_model,
                            self.n_heads,
                        )._build(),
                        self.d_model,
                        self.c_out,
                        self.d_ff,
                        moving_avg=self.moving_avg,
                        dropout=self.dropout,
                        activation=self.activation,
                    )
                    for _ in range(self.d_layers)._build()
                ],
                norm_layer=LTSFLayerNorm(self.d_model)._build(),
                projection=nn.Linear(self.d_model, self.c_out, bias=True),
            )._build()

        def forward(
            self,
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
        ):
            from torch import cat, mean, zeros

            # decomp init
            _mean = mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            _zeros = zeros(
                [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
            )
            seasonal_init, trend_init = self.decomp(x_enc)
            # decoder input
            trend_init = cat([trend_init[:, -self.label_len :, :], _mean], dim=1)
            seasonal_init = cat([seasonal_init[:, -self.label_len :, :], _zeros], dim=1)
            # enc
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            # dec
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(
                dec_out,
                enc_out,
                x_mask=dec_self_mask,
                cross_mask=dec_enc_mask,
                trend=trend_init,
            )
            # final
            dec_out = trend_part + seasonal_part

            if self.output_attention:
                return dec_out[:, -self.pred_len :, :], attns
            else:
                return dec_out[:, -self.pred_len :, :]  # [B, L, D]
