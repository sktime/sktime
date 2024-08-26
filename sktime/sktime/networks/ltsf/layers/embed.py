"""Extra LTSF Model Layers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


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


class LTSFTemporalEmbedding:
    """LTSFTemporalEmbedding."""  # combines [] from cure-lab

    def __init__(self, temporal_encoding_type, mark_vocab_sizes, d_model):
        self.temporal_encoding_type = temporal_encoding_type
        self.mark_vocab_sizes = mark_vocab_sizes
        self.d_model = d_model

    def _build(self):
        return self._LTSFTemporalEmbedding(
            self.temporal_encoding_type, self.mark_vocab_sizes, self.d_model
        )

    class _LTSFTemporalEmbedding(nn_module):
        def __init__(self, temporal_encoding_type, mark_vocab_sizes, d_model):
            super().__init__()

            self.temporal_encoding_type = temporal_encoding_type

            if (
                temporal_encoding_type == "embed"
                or temporal_encoding_type == "fixed-embed"
            ):
                self.embeds = nn.ModuleList()
                for num_embeddings in mark_vocab_sizes:
                    embed = nn.Embedding(
                        num_embeddings=num_embeddings, embedding_dim=d_model
                    )

                    if temporal_encoding_type == "fixed-embed":
                        # fix the embedding layer
                        w = torch.zeros(num_embeddings, d_model).float()
                        w.require_grad = False

                        position = torch.arange(0, num_embeddings).float().unsqueeze(1)
                        div_term = (
                            torch.arange(0, d_model, 2).float()
                            * -(math.log(10000.0) / d_model)
                        ).exp()

                        w[:, 0::2] = torch.sin(position * div_term)
                        w[:, 1::2] = torch.cos(position * div_term)

                        embed.weight = nn.Parameter(w, requires_grad=False)

                    self.embeds.append(embed)

            else:
                # linear or any other transformation
                self.linear = nn.Linear(
                    in_features=len(mark_vocab_sizes),
                    out_features=d_model,
                    bias=False,
                    dtype=torch.float,
                )

        def forward(self, x):
            if (
                self.temporal_encoding_type == "embed"
                or self.temporal_encoding_type == "fixed-embed"
            ):
                x = x.long()
                result = 0
                for i, embed in enumerate(self.embeds):
                    result += embed(x[:, :, i])
            else:
                result = self.linear(x)

            return result


class LTSFDataEmbedding:
    """LTSFDataEmbedding."""

    def __init__(
        self,
        in_channels,
        d_model,
        dropout=0.1,
        mark_vocab_sizes=None,
        position_encoding=True,
        temporal_encoding=True,
        temporal_encoding_type="linear",
    ):
        self.in_channels = in_channels
        self.d_model = d_model
        self.dropout = dropout
        self.mark_vocab_sizes = mark_vocab_sizes
        self.temporal_encoding = temporal_encoding
        self.position_encoding = position_encoding
        self.temporal_encoding_type = temporal_encoding_type

    def _build(self):
        return self._LTSFDataEmbedding(
            self.in_channels,
            self.d_model,
            self.dropout,
            self.mark_vocab_sizes,
            self.position_encoding,
            self.temporal_encoding,
            self.temporal_encoding_type,
        )

    class _LTSFDataEmbedding(nn_module):
        def __init__(
            self,
            in_channels,
            d_model,
            dropout=0.1,
            mark_vocab_sizes=None,
            temporal_encoding_type="linear",
            position_encoding=True,
            temporal_encoding=True,
        ):
            super().__init__()

            self.position_encoding = position_encoding
            self.temporal_encoding = temporal_encoding

            self.value_embedding = LTSFTokenEmbedding(
                in_channels=in_channels, d_model=d_model
            )._build()

            if position_encoding:
                self.position_embedding = LTSFPositionalEmbedding(
                    d_model=d_model
                )._build()

            if temporal_encoding:
                self.temporal_embedding = LTSFTemporalEmbedding(
                    temporal_encoding_type=temporal_encoding_type,
                    mark_vocab_sizes=mark_vocab_sizes,
                    d_model=d_model,
                )._build()

            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            out = self.value_embedding(x)

            if self.position_encoding:
                out += self.position_embedding(x)

            if self.temporal_encoding:
                out += self.temporal_embedding(x_mark)

            return self.dropout(out)
