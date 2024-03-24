"""Extra LTSF-Linear Model Layers."""
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

        pass


import math

import numpy as np


class SeriesDecomposer:
    """Series decomposition block."""

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def _build(self):
        return self._SeriesDecomposer(self.kernel_size)

    class _SeriesDecomposer(nn_module):
        """Series decomposition block."""

        def __init__(self, kernel_size):
            super().__init__()
            self.moving_avg = MovingAverage(kernel_size, stride=1)._build()

        def forward(self, x):
            moving_mean = self.moving_avg(x)
            res = x - moving_mean
            return res, moving_mean


class MovingAverage:
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def _build(self):
        return self._MovingAverage(self.kernel_size, self.stride)

    class _MovingAverage(nn_module):
        """Moving average block to highlight the trend of time series."""

        def __init__(self, kernel_size, stride):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(self, x):
            from torch import cat

            # padding on the both ends of time series
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            return x


class AutoCorrelation:
    """AutoCorrelation layer.

    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    class _AutoCorrelation(nn_module):
        def __init__(
            self,
            mask_flag=True,
            factor=1,
            scale=None,
            attention_dropout=0.1,
            output_attention=False,
        ):
            super().__init__()
            self.factor = factor
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)

        def time_delay_agg_training(self, values, corr):
            """Normalize.

            SpeedUp version of Autocorrelation (a batch-normalization style design)
            This is for the training phase.
            """
            from torch import mean, roll, softmax, stack, topk, zeros_like

            head = values.shape[1]
            channel = values.shape[2]
            length = values.shape[3]
            # find top k
            top_k = int(self.factor * math.log(length))
            mean_value = mean(mean(corr, dim=1), dim=1)
            index = topk(mean(mean_value, dim=0), top_k, dim=-1)[1]
            weights = stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
            # update corr
            tmp_corr = softmax(weights, dim=-1)
            # aggregation
            tmp_values = values
            delays_agg = zeros_like(values).float()
            for i in range(top_k):
                pattern = roll(tmp_values, -int(index[i]), -1)
                delays_agg = delays_agg + pattern * (
                    tmp_corr[:, i]
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, head, channel, length)
                )
            return delays_agg

        def time_delay_agg_inference(self, values, corr):
            """Normalize data.

            SpeedUp version of Autocorrelation (a batch-normalization style design)
            This is for the inference phase.
            """
            from torch import arange, gather, mean, softmax, topk, zeros_like

            batch = values.shape[0]
            head = values.shape[1]
            channel = values.shape[2]
            length = values.shape[3]
            # index init
            init_index = (
                arange(length)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch, head, channel, 1)
                .cuda()
            )
            # find top k
            top_k = int(self.factor * math.log(length))
            mean_value = mean(mean(corr, dim=1), dim=1)
            weights = topk(mean_value, top_k, dim=-1)[0]
            delay = topk(mean_value, top_k, dim=-1)[1]
            # update corr
            tmp_corr = softmax(weights, dim=-1)
            # aggregation
            tmp_values = values.repeat(1, 1, 1, 2)
            delays_agg = zeros_like(values).float()
            for i in range(top_k):
                tmp_delay = init_index + (
                    delay[:, i]
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, head, channel, length)
                )
                pattern = gather(tmp_values, dim=-1, index=tmp_delay)
                delays_agg = delays_agg + pattern * (
                    tmp_corr[:, i]
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, head, channel, length)
                )
            return delays_agg

        def time_delay_agg_full(self, values, corr):
            """Normalize Data.

            Standard version of Autocorrelation
            """
            from torch import arange, gather, softmax, topk, zeros_like

            batch = values.shape[0]
            head = values.shape[1]
            channel = values.shape[2]
            length = values.shape[3]
            # index init
            init_index = (
                arange(length)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch, head, channel, 1)
                .cuda()
            )
            # find top k
            top_k = int(self.factor * math.log(length))
            weights = topk(corr, top_k, dim=-1)[0]
            delay = topk(corr, top_k, dim=-1)[1]
            # update corr
            tmp_corr = softmax(weights, dim=-1)
            # aggregation
            tmp_values = values.repeat(1, 1, 1, 2)
            delays_agg = zeros_like(values).float()
            for i in range(top_k):
                tmp_delay = init_index + delay[..., i].unsqueeze(-1)
                pattern = gather(tmp_values, dim=-1, index=tmp_delay)
                delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
            return delays_agg

        def forward(self, queries, keys, values, attn_mask):
            """Call model."""
            from torch import cat, conj, fft, zeros_like

            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            if L > S:
                zeros = zeros_like(queries[:, : (L - S), :]).float()
                values = cat([values, zeros], dim=1)
                keys = cat([keys, zeros], dim=1)
            else:
                values = values[:, :L, :, :]
                keys = keys[:, :L, :, :]

            # period-based dependencies
            q_fft = fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
            k_fft = fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
            res = q_fft * conj(k_fft)
            corr = fft.irfft(res, dim=-1)

            # time delay agg
            if self.training:
                V = self.time_delay_agg_training(
                    values.permute(0, 2, 3, 1).contiguous(), corr
                ).permute(0, 3, 1, 2)
            else:
                V = self.time_delay_agg_inference(
                    values.permute(0, 2, 3, 1).contiguous(), corr
                ).permute(0, 3, 1, 2)

            if self.output_attention:
                return (V.contiguous(), corr.permute(0, 3, 1, 2))
            else:
                return (V.contiguous(), None)

    def __init__(
        self,
        mask_flag=True,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention

    def _build(self):
        return self._AutoCorrelation(
            self.mask_flag,
            self.factor,
            self.scale,
            self.attention_dropout,
            self.output_attention,
        )


class AutoCorrelationLayer:
    """Layer for AutoCorrelation."""

    class _AutoCorrelationLayer(nn_module):
        def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
            super().__init__()

            d_keys = d_keys or (d_model // n_heads)
            d_values = d_values or (d_model // n_heads)

            self.inner_correlation = correlation
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
            self.out_projection = nn.Linear(d_values * n_heads, d_model)
            self.n_heads = n_heads

        def forward(self, queries, keys, values, attn_mask):
            """Call layer."""
            B, L, _ = queries.shape
            _, S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values).view(B, S, H, -1)

            out, attn = self.inner_correlation(queries, keys, values, attn_mask)
            out = out.view(B, L, -1)

            return self.out_projection(out), attn

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        self.correlation = correlation
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def _build(self):
        return self._AutoCorrelationLayer(
            self.correlation, self.d_model, self.n_heads, self.d_keys, self.d_values
        )


class LTSFLayerNorm:
    """LayerNorm."""

    class _LTSFLayernorm(nn_module):
        def __init__(self, channels):
            super().__init__()
            self.layernorm = nn.LayerNorm(channels)

        def forward(self, x):
            """Call layer."""
            from torch import mean

            x_hat = self.layernorm(x)
            bias = mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
            return x_hat - bias

    def __init__(self, channels):
        self.channels = channels

    def _build(self):
        return self._LTSFLayernorm(self.channels)


class EncoderLayer:
    """Encoder Layer.

    Autoformer encoder layer with the progressive decomposition architecture
    """

    class _EncoderLayer(nn_module):
        def __init__(
            self,
            attention,
            d_model,
            d_ff=None,
            moving_avg=25,
            dropout=0.1,
            activation="relu",
        ):
            super().__init__()
            from torch.nn.functional import gelu, relu

            d_ff = d_ff or 4 * d_model
            self.attention = attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
            )
            self.decomp1 = SeriesDecomposer(moving_avg)
            self.decomp2 = SeriesDecomposer(moving_avg)
            self.dropout = nn.Dropout(dropout)
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, attn_mask=None):
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)
            x, _ = self.decomp1(x)
            y = x
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            res, _ = self.decomp2(x + y)
            return res, attn

    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._EncoderLayer(
            self.attention,
            self.d_model,
            self.d_ff,
            self.moving_avg,
            self.dropout,
            self.activation,
        )


class Encoder:
    """Autoformer encoder."""

    class _Encoder(nn_module):
        def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
            super(Encoder, self).__init__()
            self.attn_layers = nn.ModuleList(attn_layers)
            self.conv_layers = (
                nn.ModuleList(conv_layers) if conv_layers is not None else None
            )
            self.norm = norm_layer

        def forward(self, x, attn_mask=None):
            """Call encoder."""
            attns = []
            if self.conv_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    attns.append(attn)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers
        self.norm_layer = norm_layer

    def _build(self):
        return self._Encoder(self.attn_layers, self.conv_layers, self.norm_layer)


class DecoderLayer:
    """Decoder Layer.

    Autoformer decoder layer with the progressive decomposition architecture
    """

    class _DecoderLayer(nn_module):
        def __init__(
            self,
            self_attention,
            cross_attention,
            d_model,
            c_out,
            d_ff=None,
            moving_avg=25,
            dropout=0.1,
            activation="relu",
        ):
            super().__init__()
            from torch.nn.functional import gelu, relu

            d_ff = d_ff or 4 * d_model
            self.self_attention = self_attention
            self.cross_attention = cross_attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
            )
            self.decomp1 = SeriesDecomposer(moving_avg)
            self.decomp2 = SeriesDecomposer(moving_avg)
            self.decomp3 = SeriesDecomposer(moving_avg)
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Conv1d(
                in_channels=d_model,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
                bias=False,
            )
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, cross, x_mask=None, cross_mask=None):
            """Call decoder."""
            x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
            x, trend1 = self.decomp1(x)
            x = x + self.dropout(
                self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
            )
            x, trend2 = self.decomp2(x)
            y = x
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            x, trend3 = self.decomp3(x + y)

            residual_trend = trend1 + trend2 + trend3
            residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(
                1, 2
            )
            return x, residual_trend

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.c_out = c_out
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._DecoderLayer(
            self.self_attention,
            self.cross_attention,
            self.d_model,
            self.c_out,
            self.d_ff,
            self.moving_avg,
            self.dropout,
            self.activation,
        )


class Decoder:
    """Autoformer decoder."""

    class _Decoder(nn_module):
        def __init__(self, layers, norm_layer=None, projection=None):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.norm = norm_layer
            self.projection = projection

        def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
            """Call decoder."""
            for layer in self.layers:
                x, residual_trend = layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask
                )
                trend = trend + residual_trend

            if self.norm is not None:
                x = self.norm(x)

            if self.projection is not None:
                x = self.projection(x)
            return x, trend

    def __init__(self, layers, norm_layer=None, projection=None):
        self.layers = layers
        self.norm_layer = norm_layer
        self.projection = projection

    def _build(self):
        return self._Decoder(self.layers, self.norm_layer, self.projection)


class LTSFTriangularCausalMask:
    """LTSFTriangularCausalMask."""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        """mask."""
        return self._mask


class LTSFProbMask:
    """LTSFProbMask."""

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """mask."""
        return self._mask


class LTSFPositionalEmbedding:
    """LTSFPositionalEmbedding."""

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def _build(self):
        return self._LTSFPositionalEmbedding(self.d_model, self.max_len)

    class _LTSFPositionalEmbedding(nn_module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model).float()
            pe.require_grad = False

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return self.pe[:, : x.size(1)]


class LTSFTokenEmbedding:
    """LTSFTokenEmbedding."""

    def __init__(self, in_channels, d_model):
        self.in_channels = in_channels
        self.d_model = d_model

    def _build(self):
        return self._LTSFTokenEmbedding(self.in_channels, self.d_model)

    class _LTSFTokenEmbedding(nn_module):
        def __init__(self, in_channels, d_model):
            super().__init__()
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            self.tokenConv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=3,
                padding=padding,
                padding_mode="circular",
                bias=False,
            )
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="leaky_relu"
                    )

        def forward(self, x):
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
            return x


class LTSFFixedEmbedding:
    """LTSFFixedEmbedding."""

    def __init__(self, in_channels, d_model):
        self.in_channels = in_channels
        self.d_model = d_model

    def _build(self):
        return self._LTSFFixedEmbedding(self.in_channels, self.d_model)

    class _LTSFFixedEmbedding(nn_module):
        def __init__(self, in_channels, d_model):
            super().__init__()

            w = torch.zeros(in_channels, d_model).float()
            w.require_grad = False

            position = torch.arange(0, in_channels).float().unsqueeze(1)
            div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            w[:, 0::2] = torch.sin(position * div_term)
            w[:, 1::2] = torch.cos(position * div_term)

            self.emb = nn.Embedding(in_channels, d_model)
            self.emb.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return self.emb(x).detach()


class LTSFTemporalEmbedding:
    """LTSFTemporalEmbedding."""

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

    def _build(self):
        return self._LTSFTemporalEmbedding(self.d_model, self.embed_type, self.freq)

    class _LTSFTemporalEmbedding(nn_module):
        def __init__(self, d_model, embed_type="fixed", freq="h"):
            super().__init__()

            minute_size = 4
            hour_size = 24
            weekday_size = 7
            day_size = 32
            month_size = 13

            if embed_type == "fixed":
                if freq == "t":
                    self.minute_embed = LTSFFixedEmbedding(
                        minute_size, d_model
                    )._build()
                self.hour_embed = LTSFFixedEmbedding(hour_size, d_model)._build()
                self.weekday_embed = LTSFFixedEmbedding(weekday_size, d_model)._build()
                self.day_embed = LTSFFixedEmbedding(day_size, d_model)._build()
                self.month_embed = LTSFFixedEmbedding(month_size, d_model)._build()
            else:
                if freq == "t":
                    self.minute_embed = nn.Embedding(minute_size, d_model)
                self.hour_embed = nn.Embedding(hour_size, d_model)
                self.weekday_embed = nn.Embedding(weekday_size, d_model)
                self.day_embed = nn.Embedding(day_size, d_model)
                self.month_embed = nn.Embedding(month_size, d_model)

        def forward(self, x):
            x = x.long()

            minute_x = (
                self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
            )
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            day_x = self.day_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 0])

            return hour_x + weekday_x + day_x + month_x + minute_x


class LTSFTimeFeatureEmbedding:
    """LTSFTimeFeatureEmbedding."""

    def __init__(self, d_model, embed_type="timeF", freq="h"):
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

    def _build(self):
        return self._LTSFTimeFeatureEmbedding(self.d_model, self.embed_type, self.freq)

    class _LTSFTimeFeatureEmbedding(nn_module):
        def __init__(self, d_model, embed_type="timeF", freq="h"):
            super().__init__()

            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model, bias=False)

        def forward(self, x):
            return self.embed(x)


class LTSFDataEmbedding:
    """LTSFDataEmbedding."""

    def __init__(self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.in_channels = in_channels
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._LTSFDataEmbedding(
            self.in_channels, self.d_model, self.embed_type, self.freq, self.dropout
        )

    class _LTSFDataEmbedding(nn_module):
        def __init__(
            self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1
        ):
            super().__init__()

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()
            self.position_embedding = LTSFPositionalEmbedding(d_model=d_model)._build()
            self.temporal_embedding = (
                LTSFTemporalEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
                if embed_type != "timeF"
                else LTSFTimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
            return self.dropout(x)


class LTSFDataEmbeddingWOPos:
    """LTSFDataEmbeddingWOPos."""

    def __init__(self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.in_channels = in_channels
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._LTSFDataEmbeddingWOPos(
            self.in_channels, self.d_model, self.embed_type, self.freq, self.dropout
        )

    class _LTSFDataEmbeddingWOPos(nn_module):
        def __init__(
            self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1
        ):
            super().__init__()

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()
            self.position_embedding = LTSFPositionalEmbedding(d_model=d_model)._build()
            self.temporal_embedding = (
                LTSFTemporalEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
                if embed_type != "timeF"
                else LTSFTimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
            return self.dropout(x)


class LTSFDataEmbeddingWOPosTemp:
    """LTSFDataEmbeddingWOPosTemp."""

    def __init__(self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.in_channels = in_channels
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._LTSFDataEmbeddingWOPosTemp(
            self.in_channels, self.d_model, self.embed_type, self.freq, self.dropout
        )

    class _LTSFDataEmbeddingWOPosTemp(nn_module):
        def __init__(
            self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1
        ):
            super().__init__()

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()
            self.position_embedding = LTSFPositionalEmbedding(d_model=d_model)._build()
            self.temporal_embedding = (
                LTSFTemporalEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
                if embed_type != "timeF"
                else LTSFTimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            x = self.value_embedding(x)
            return self.dropout(x)


class LTSFDataEmbeddingWOTemp:
    """LTSFDataEmbeddingWOTemp."""

    def __init__(self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.in_channels = in_channels
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._LTSFDataEmbeddingWOTemp(
            self.in_channels, self.d_model, self.embed_type, self.freq, self.dropout
        )

    class _LTSFDataEmbeddingWOTemp(nn_module):
        def __init__(
            self, in_channels, d_model, embed_type="fixed", freq="h", dropout=0.1
        ):
            super().__init__()

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()
            self.position_embedding = LTSFPositionalEmbedding(d_model=d_model)._build()
            self.temporal_embedding = (
                LTSFTemporalEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
                if embed_type != "timeF"
                else LTSFTimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )._build()
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            x = self.value_embedding(x) + self.position_embedding(x)
            return self.dropout(x)


class LTSFFullAttention:
    """LTSFFullAttention."""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention

    def _build(self):
        return self._LTSFFullAttention(
            self.mask_flag,
            self.factor,
            self.scale,
            self.attention_dropout,
            self.output_attention,
        )

    class _LTSFFullAttention(nn_module):
        def __init__(
            self,
            mask_flag=True,
            factor=5,
            scale=None,
            attention_dropout=0.1,
            output_attention=False,
        ):
            super().__init__()
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)

        def forward(self, queries, keys, values, attn_mask):
            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            scale = self.scale or 1.0 / math.sqrt(E)

            scores = torch.einsum("blhe,bshe->bhls", queries, keys)

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = LTSFTriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

            if self.output_attention:
                return (V.contiguous(), A)
            else:
                return (V.contiguous(), None)


class LTSFProbAttention:
    """LTSFProbAttention."""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention

    def _build(self):
        return self._LTSFProbAttention(
            self.mask_flag,
            self.factor,
            self.scale,
            self.attention_dropout,
            self.output_attention,
        )

    class _LTSFProbAttention(nn_module):
        def __init__(
            self,
            mask_flag=True,
            factor=5,
            scale=None,
            attention_dropout=0.1,
            output_attention=False,
        ):
            super().__init__()
            self.factor = factor
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)

        def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
            # Q [B, H, L, D]
            B, H, L_K, E = K.shape
            _, _, L_Q, _ = Q.shape

            # calculate the sampled Q_K
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
            index_sample = torch.randint(
                L_K, (L_Q, sample_k)
            )  # real U = U_part(factor*ln(L_k))*L_q
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
            Q_K_sample = torch.matmul(
                Q.unsqueeze(-2), K_sample.transpose(-2, -1)
            ).squeeze()

            # find the Top_k query with sparisty measurement
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
            M_top = M.topk(n_top, sorted=False)[1]

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
            ]  # factor*ln(L_q)
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

            return Q_K, M_top

        def _get_initial_context(self, V, L_Q):
            B, H, L_V, D = V.shape
            if not self.mask_flag:
                # V_sum = V.sum(dim=-2)
                V_sum = V.mean(dim=-2)
                contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
            else:  # use mask
                assert (
                    L_Q == L_V
                )  # requires that L_Q == L_V, i.e. for self-attention only
                contex = V.cumsum(dim=-2)
            return contex

        def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
            B, H, L_V, D = V.shape

            if self.mask_flag:
                attn_mask = LTSFProbMask(B, H, L_Q, index, scores, device=V.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)

            attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

            context_in[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = torch.matmul(attn, V).type_as(context_in)
            if self.output_attention:
                attns = (
                    (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
                )
                attns[
                    torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index,
                    :,
                ] = attn
                return (context_in, attns)
            else:
                return (context_in, None)

        def forward(self, queries, keys, values, attn_mask):
            B, L_Q, H, D = queries.shape
            _, L_K, _, _ = keys.shape

            queries = queries.transpose(2, 1)
            keys = keys.transpose(2, 1)
            values = values.transpose(2, 1)

            U_part = (
                self.factor * np.ceil(np.log(L_K)).astype("int").item()
            )  # c*ln(L_k)
            u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q

            scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

            # add scale factor
            scale = self.scale or 1.0 / math.sqrt(D)
            if scale is not None:
                scores_top = scores_top * scale
            # get the context
            context = self._get_initial_context(values, L_Q)
            # update the context with selected top_k queries
            context, attn = self._update_context(
                context, values, scores_top, index, L_Q, attn_mask
            )

            return context.contiguous(), attn


class LTSFAttentionLayer:
    """LTSFAttentionLayer."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        self.attention = attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def _build(self):
        return self._LTSFAttentionLayer(
            self.attention, self.d_model, self.n_heads, self.d_keys, self.d_values
        )

    class _LTSFAttentionLayer(nn_module):
        def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
            super().__init__()

            d_keys = d_keys or (d_model // n_heads)
            d_values = d_values or (d_model // n_heads)

            self.inner_attention = attention
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
            self.out_projection = nn.Linear(d_values * n_heads, d_model)
            self.n_heads = n_heads

        def forward(self, queries, keys, values, attn_mask):
            B, L, _ = queries.shape
            _, S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values).view(B, S, H, -1)

            out, attn = self.inner_attention(queries, keys, values, attn_mask)
            out = out.view(B, L, -1)

            return self.out_projection(out), attn


class LTSFTransformerConvLayer:
    """LTSFTransformerConvLayer."""

    def __init__(self, in_channels):
        self.in_channels = in_channels

    def _build(self):
        return self._LTSFTransformerConvLayer(self.in_channels)

    class _LTSFTransformerConvLayer(nn_module):
        def __init__(self, in_channels):
            super().__init__()
            self.downConv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=2,
                padding_mode="circular",
            )
            self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.ELU()
            self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.downConv(x.permute(0, 2, 1))
            x = self.norm(x)
            x = self.activation(x)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x


class LTSFTransformerEncoderLayer:
    """LTSFTransformerEncoderLayer."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._LTSFTransformerEncoderLayer(
            self.attention, self.d_model, self.d_ff, self.dropout, self.activation
        )

    class _LTSFTransformerEncoderLayer(nn_module):
        def __init__(
            self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"
        ):
            from torch.nn.functional import gelu, relu

            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attention = attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, attn_mask=None):
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)

            y = x = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm2(x + y), attn


class LTSFTransformerEncoder:
    """LTSFTransformerEncoder."""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers
        self.norm_layer = norm_layer

    def _build(self):
        return self._LTSFTransformerEncoder(
            self.attn_layers, self.conv_layers, self.norm_layer
        )

    class _LTSFTransformerEncoder(nn_module):
        def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
            super().__init__()
            self.attn_layers = nn.ModuleList(attn_layers)
            self.conv_layers = (
                nn.ModuleList(conv_layers) if conv_layers is not None else None
            )
            self.norm = norm_layer

        def forward(self, x, attn_mask=None):
            # x [B, L, D]
            attns = []
            if self.conv_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    attns.append(attn)

            if self.norm is not None:
                x = self.norm(x)

            return x, attns


class LTSFTransformerDecoderLayer:
    """LTSFTransformerDecoderLayer."""

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

    def _build(self):
        return self._LTSFTransformerDecoderLayer(
            self.self_attention,
            self.cross_attention,
            self.d_model,
            self.d_ff,
            self.dropout,
            self.activation,
        )

    class _LTSFTransformerDecoderLayer(nn_module):
        def __init__(
            self,
            self_attention,
            cross_attention,
            d_model,
            d_ff=None,
            dropout=0.1,
            activation="relu",
        ):
            from torch.nn.functional import gelu, relu

            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.self_attention = self_attention
            self.cross_attention = cross_attention
            self.conv1 = nn.Conv1d(
                in_channels=d_model, out_channels=d_ff, kernel_size=1
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff, out_channels=d_model, kernel_size=1
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = relu if activation == "relu" else gelu

        def forward(self, x, cross, x_mask=None, cross_mask=None):
            x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
            x = self.norm1(x)

            x = x + self.dropout(
                self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
            )

            y = x = self.norm2(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm3(x + y)


class LTSFTransformerDecoder:
    """LTSFTransformerDecoder."""

    def __init__(self, layers, norm_layer=None, projection=None):
        self.layers = layers
        self.norm_layer = norm_layer
        self.projection = projection

    def _build(self):
        return self._LTSFTransformerDecoder(
            self.layers, self.norm_layer, self.projection
        )

    class _LTSFTransformerDecoder(nn_module):
        def __init__(self, layers, norm_layer=None, projection=None):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.norm = norm_layer
            self.projection = projection

        def forward(self, x, cross, x_mask=None, cross_mask=None):
            for layer in self.layers:
                x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            if self.norm is not None:
                x = self.norm(x)

            if self.projection is not None:
                x = self.projection(x)
            return x
