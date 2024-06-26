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

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _build(self):
        return self._LTSFFixedEmbedding(self.num_embeddings, self.embedding_dim)

    class _LTSFFixedEmbedding(nn_module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()

            w = torch.zeros(num_embeddings, embedding_dim).float()
            w.require_grad = False

            position = torch.arange(0, num_embeddings).float().unsqueeze(1)
            div_term = (
                torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)
            ).exp()

            w[:, 0::2] = torch.sin(position * div_term)
            w[:, 1::2] = torch.cos(position * div_term)

            self.emb = nn.Embedding(num_embeddings, embedding_dim)
            self.emb.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return self.emb(x).detach()


class LTSFTemporalEmbeddingEmbed:
    """LTSFTemporalEmbedding."""

    # inside LTSFTemporalEmbedding of cure-lab

    def __init__(self, mark_vocab_sizes, d_model):
        self.mark_vocab_sizes = mark_vocab_sizes
        self.d_model = d_model

    def _build(self):
        return self._LTSFTemporalEmbedding(self.mark_vocab_sizes, self.d_model)

    class _LTSFTemporalEmbedding(nn_module):
        def __init__(self, mark_vocab_sizes, d_model):
            super().__init__()

            self.embeds = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=mark_vocab_size,
                    embedding_dim=d_model
                ) for mark_vocab_size in mark_vocab_sizes
            ])

        def forward(self, x):
            x = x.long()

            result = 0
            for i, embed in enumerate(self.embeds):
                result += embed(x[:, :, i])

            return result


class LTSFTemporalEmbeddingFixedEmbed:
    """LTSFTemporalEmbedding."""

    # inside LTSFTemporalEmbedding of cure-lab

    def __init__(self, mark_vocab_sizes, d_model):
        self.mark_vocab_sizes = mark_vocab_sizes
        self.d_model = d_model

    def _build(self):
        return self._LTSFTemporalEmbeddingFixedEmbed(self.mark_vocab_sizes, self.d_model)

    class _LTSFTemporalEmbeddingFixedEmbed(nn_module):
        def __init__(self, mark_vocab_sizes, d_model):
            super().__init__()

            self.embeds = nn.ModuleList([
                LTSFFixedEmbedding(
                    num_embeddings=mark_vocab_size,
                    embedding_dim=d_model
                )._build() for mark_vocab_size in mark_vocab_sizes
            ])

        def forward(self, x):
            x = x.long()

            result = 0
            for i, embed in enumerate(self.embeds):
                result += embed(x[:, :, i])

            return result


class LTSFTemporalEmbeddingLinear:
    """LTSFTemporalEmbedding."""

    # LTSFTimeFeatureEmbedding of cure-lab

    def __init__(self, mark_vocab_sizes, d_model):
        self.mark_vocab_sizes = mark_vocab_sizes
        self.d_model = d_model

    def _build(self):
        return self._LTSFTemporalEmbeddingLinear(self.mark_vocab_sizes, self.d_model)

    class _LTSFTemporalEmbeddingLinear(nn_module):
        def __init__(self, mark_vocab_sizes, d_model):
            super().__init__()

            in_channels = len(mark_vocab_sizes)
            self.embed = nn.Linear(in_channels, d_model, bias=False)

        def forward(self, x):
            return self.embed(x)


class LTSFDataEmbedding:
    """LTSFDataEmbedding."""

    def __init__(self, in_channels, d_model, freq="h", dropout=0.1, mark_vocab_sizes=None, position_encoding=True, temporal_encoding=True, temporal_encoding_type='linear'):
        self.in_channels = in_channels
        self.d_model = d_model
        self.freq = freq
        self.dropout = dropout
        self.mark_vocab_sizes = mark_vocab_sizes
        self.temporal_encoding = temporal_encoding
        self.position_encoding = position_encoding
        self.temporal_encoding_type = temporal_encoding_type

    def _build(self):
        return self._LTSFDataEmbedding(
            self.in_channels, self.d_model, self.freq, self.dropout, self.mark_vocab_sizes, self.position_encoding, self.temporal_encoding, self.temporal_encoding_type
        )

    class _LTSFDataEmbedding(nn_module):
        def __init__(
            self, in_channels, d_model, freq="h", dropout=0.1, mark_vocab_sizes=None, temporal_encoding_type="linear", position_encoding=True, temporal_encoding=True
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
                if temporal_encoding_type == "linear":
                    self.temporal_embedding = LTSFTemporalEmbeddingLinear(mark_vocab_sizes=mark_vocab_sizes, d_model=d_model)._build()
                elif temporal_encoding_type == "embed":
                    self.temporal_embedding = LTSFTemporalEmbeddingEmbed(mark_vocab_sizes=mark_vocab_sizes, d_model=d_model)._build()
                elif temporal_encoding_type == "fixed-embed":
                    self.temporal_embedding = LTSFTemporalEmbeddingFixedEmbed(mark_vocab_sizes=mark_vocab_sizes, d_model=d_model)._build()
                else:
                    raise ValueError()

            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):

            x = self.value_embedding(x)

            if self.position_encoding:
                x += self.position_embedding(x)

            if self.temporal_encoding:
                x += self.temporal_embedding(x_mark)

            return self.dropout(x)
