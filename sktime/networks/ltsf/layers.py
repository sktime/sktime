"""Extra LTSF-Linear Model Layers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


import math


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
