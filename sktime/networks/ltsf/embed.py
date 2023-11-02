"""Extra LTSF-Linear Model Embedding Layers."""
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

        pass


import math


class PositionalEmbedding:
    """Positional Embedding."""

    class _PositionalEmbedding(nn_module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            from torch import arange, cos, sin, zeros

            # Compute the positional encodings once in log space.
            pe = zeros(max_len, d_model).float()
            pe.require_grad = False

            position = arange(0, max_len).float().unsqueeze(1)
            div_term = (
                arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            pe[:, 0::2] = sin(position * div_term)
            pe[:, 1::2] = cos(position * div_term)

            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return self.pe[:, : x.size(1)]

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def _build(self):
        return self._PositionalEmbedding(self.d_model, self.max_len)


class TokenEmbedding:
    """Token Embedding."""

    class _TokenEmbedding(nn_module):
        def __init__(self, c_in, d_model):
            super(TokenEmbedding, self).__init__()
            padding = 2
            self.tokenConv = nn.Conv1d(
                in_channels=c_in,
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

    def __init__(self, c_in, d_model):
        self.c_in = c_in
        self.d_model = d_model

    def _build(self):
        return self._TokenEmbedding(self.c_in, self.d_model)


class FixedEmbedding:
    """Fixed Embedding."""

    class _FixedEmbedding(nn_module):
        def __init__(self, c_in, d_model):
            super().__init__()

            from torch import arange, cos, sin, zeros

            w = zeros(c_in, d_model).float()
            w.require_grad = False

            position = arange(0, c_in).float().unsqueeze(1)
            div_term = (
                arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            w[:, 0::2] = sin(position * div_term)
            w[:, 1::2] = cos(position * div_term)

            self.emb = nn.Embedding(c_in, d_model)
            self.emb.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return self.emb(x).detach()

    def __init__(self, c_in, d_model):
        self.c_in = c_in
        self.d_model = d_model

    def _build(self):
        return self._FixedEmbedding(self.c_in, self.d_model)


class TemporalEmbedding:
    """Temporal Embedding."""

    class _TemporalEmbedding(nn_module):
        def __init__(self, d_model, embed_type="fixed", freq="h"):
            super().__init__()

            minute_size = 4
            hour_size = 24
            weekday_size = 7
            day_size = 32
            month_size = 13

            Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
            if freq == "t":
                self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
            self.day_embed = Embed(day_size, d_model)
            self.month_embed = Embed(month_size, d_model)

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

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

    def _build(self):
        return self._TemporalEmbedding(self.d_model, self.embed_type, self.freq)


class TimeFeatureEmbedding:
    """Time Feature Embedding."""

    class _TimeFeatureEmbedding(nn_module):
        def __init__(self, d_model, freq="h"):
            super().__init__()

            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model, bias=False)

        def forward(self, x):
            return self.embed(x)

    def __init__(self, d_model, freq="h"):
        self.d_model = d_model
        self.freq = freq

    def _build(self):
        return self._TimeFeatureEmbedding(self.d_model, self.freq)


class DataEmbedding:
    """Embedding class."""

    class _DataEmbedding(nn_module):
        def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
            super().__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = (
                TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                if embed_type != "timeF"
                else TimeFeatureEmbedding(d_model=d_model, freq=freq)
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x)
                + self.position_embedding(x)
            )
            return self.dropout(x)

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.c_in = c_in
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._DataEmbedding(
            self.c_in, self.d_model, self.embed_type, self.freq, self.dropout
        )


class DataEmbeddingWithoutPositional:
    """Embedding without Positional."""

    class _DataEmbeddingWithoutPositional(nn_module):
        def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
            super().__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.temporal_embedding = (
                TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                if embed_type != "timeF"
                else TimeFeatureEmbedding(d_model=d_model, freq=freq)
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.value_embedding(x) + self.temporal_embedding(x)
            return self.dropout(x)

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.c_in = c_in
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._DataEmbeddingWithoutPositional(
            self.c_in, self.d_model, self.embed_type, self.freq, self.dropout
        )


class DataEmbeddingWithoutPositionalTemporal:
    """Embedding without Positional or Temporal."""

    class _DataEmbeddingWithoutPositionalTemporal(nn_module):
        def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
            super().__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.value_embedding(x)
            return self.dropout(x)

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.c_in = c_in
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._DataEmbeddingWithoutPositionalTemporal(
            self.c_in, self.d_model, self.embed_type, self.freq, self.dropout
        )


class DataEmbeddingWithoutTemporal:
    """Embedding without Temporal."""

    class _DataEmbeddingWithoutTemporal(nn_module):
        def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
            super().__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.value_embedding(x) + self.position_embedding(x)
            return self.dropout(x)

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        self.c_in = c_in
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout = dropout

    def _build(self):
        return self._DataEmbeddingWithoutTemporal(
            self.c_in, self.d_model, self.embed_type, self.freq, self.dropout
        )
