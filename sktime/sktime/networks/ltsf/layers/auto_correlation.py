"""Extra LTSF Model Layers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


import math


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
