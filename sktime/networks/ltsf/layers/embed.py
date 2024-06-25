"""Extra LTSF Model Layers."""
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""

        pass


import math


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

    def __init__(self, d_model, fixed_embedding=False, freq="h"):
        self.d_model = d_model
        self.fixed_embedding = fixed_embedding
        self.freq = freq

    def _build(self):
        return self._LTSFTemporalEmbedding(self.d_model, self.fixed_embedding, self.freq)

    class _LTSFTemporalEmbedding(nn_module):
        def __init__(self, d_model, fixed_embedding=False, freq="h"):
            super().__init__()

            minute_size = 4
            hour_size = 24
            weekday_size = 7
            day_size = 32
            month_size = 13

            if fixed_embedding:
                if freq == "t":
                    self.minute_embed = LTSFFixedEmbedding(minute_size, d_model)._build()
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

    def __init__(self, d_model, fixed_embedding=False, freq="h"):
        self.d_model = d_model
        self.fixed_embedding = fixed_embedding
        self.freq = freq

    def _build(self):
        return self._LTSFTimeFeatureEmbedding(self.d_model, self.fixed_embedding, self.freq)

    class _LTSFTimeFeatureEmbedding(nn_module):
        def __init__(self, d_model, fixed_embedding=False, freq="h"):
            super().__init__()

            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model, bias=False)

        def forward(self, x):
            return self.embed(x)


class LTSFDataEmbedding:
    """LTSFDataEmbedding."""

    def __init__(self, in_channels, d_model, freq="h", dropout=0.1, fixed_embedding=False, position_encoding=True, temporal_encoding=True):
        self.in_channels = in_channels
        self.d_model = d_model
        self.freq = freq
        self.dropout = dropout
        self.fixed_embedding = fixed_embedding
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding

    def _build(self):
        return self._LTSFDataEmbedding(
            self.in_channels, self.d_model, self.freq, self.dropout, self.fixed_embedding, self.position_encoding, self.temporal_encoding
        )

    class _LTSFDataEmbedding(nn_module):
        def __init__(
            self, in_channels, d_model, freq="h", dropout=0.1, fixed_embedding=False, position_encoding=True, temporal_encoding=True
        ):
            super().__init__()

            self.position_encoding = position_encoding
            self.temporal_encoding = temporal_encoding

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()

            if position_encoding:
                self.position_embedding = LTSFPositionalEmbedding(d_model=d_model)._build()

            if temporal_encoding:
                if fixed_embedding:
                    self.temporal_embedding = LTSFTemporalEmbedding(
                        d_model=d_model, fixed_embedding=fixed_embedding, freq=freq
                    )._build()
                else:
                    self.temporal_embedding = LTSFTimeFeatureEmbedding(
                        d_model=d_model, fixed_embedding=fixed_embedding, freq=freq
                    )._build()

            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):

            x = self.value_embedding(x)

            if self.position_encoding:
                x += self.position_embedding(x)

            if self.temporal_encoding:
                x += self.temporal_embedding(x_mark)

            return self.dropout(x)
