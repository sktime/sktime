"""Embeddings file for momentfm."""

import math
import warnings

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.libs.momentfm.utils.masking import Masking

if _check_soft_dependencies(["torch"], severity="none"):
    import torch
    import torch.nn as nn

    class PositionalEmbedding(nn.Module):
        """Positional Embedding."""

        def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
            super().__init__()
            self.model_name = model_name

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
            """Forward function."""
            if (
                self.model_name == "MOMENT"
                or self.model_name == "TimesNet"
                or self.model_name == "GPT4TS"
            ):
                return self.pe[:, : x.size(2)]
            else:
                return self.pe[:, : x.size(1)]

    class TokenEmbedding(nn.Module):
        """Token Embeddings."""

        def __init__(self, c_in, d_model):
            super().__init__()
            padding = 1 if torch.__version__ >= "1.5.0" else 2
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
            """Forward Function."""
            # x = x.permute(0, 2, 1)
            x = self.tokenConv(x)
            x = x.transpose(1, 2)
            # batch_size x seq_len x d_model
            return x

    class FixedEmbedding(nn.Module):
        """Fixed Embeddings."""

        def __init__(self, c_in, d_model):
            super().__init__()

            w = torch.zeros(c_in, d_model).float()
            w.require_grad = False

            position = torch.arange(0, c_in).float().unsqueeze(1)
            div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

            w[:, 0::2] = torch.sin(position * div_term)
            w[:, 1::2] = torch.cos(position * div_term)

            self.emb = nn.Embedding(c_in, d_model)
            self.emb.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            """Forward Function."""
            return self.emb(x).detach()

    class TemporalEmbedding(nn.Module):
        """Temporal Embedding."""

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
            """Forward Function."""
            x = x.long()
            minute_x = (
                self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
            )
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            day_x = self.day_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 0])

            return hour_x + weekday_x + day_x + month_x + minute_x

    class TimeFeatureEmbedding(nn.Module):
        """Time Feature Embedding."""

        def __init__(self, d_model, embed_type="timeF", freq="h"):
            super().__init__()

            freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model, bias=False)

        def forward(self, x):
            """Forward Function."""
            return self.embed(x)

    class DataEmbedding(nn.Module):
        """Data Embeddings."""

        def __init__(
            self, c_in, d_model, model_name, embed_type="fixed", freq="h", dropout=0.1
        ):
            super().__init__()
            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(
                d_model=d_model, model_name=model_name
            )
            self.temporal_embedding = (
                TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                if embed_type != "timeF"
                else TimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark=None):
            """Forward Function."""
            if x_mark is None:
                x = self.value_embedding(x) + self.position_embedding(x)
            else:
                x = (
                    self.value_embedding(x)
                    + self.temporal_embedding(x_mark)
                    + self.position_embedding(x)
                )
            return self.dropout(x)

    class DataEmbedding_wo_pos(nn.Module):
        """Data Embedding without positioning."""

        def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
            super().__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = (
                TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
                if embed_type != "timeF"
                else TimeFeatureEmbedding(
                    d_model=d_model, embed_type=embed_type, freq=freq
                )
            )
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            """Forward function."""
            if x_mark is None:
                x = self.value_embedding(x)
            else:
                x = self.value_embedding(x) + self.temporal_embedding(x_mark)
            return self.dropout(x)

    class PatchEmbedding(nn.Module):
        """Patch Embedding."""

        def __init__(
            self,
            d_model: int = 768,
            seq_len: int = 512,
            patch_len: int = 8,
            stride: int = 8,
            dropout: int = 0.1,
            add_positional_embedding: bool = False,
            value_embedding_bias: bool = False,
            orth_gain: float = 1.41,
        ):
            super().__init__()
            self.patch_len = patch_len
            self.seq_len = seq_len
            self.stride = stride
            self.d_model = d_model
            self.add_positional_embedding = add_positional_embedding

            self.value_embedding = nn.Linear(
                patch_len, d_model, bias=value_embedding_bias
            )
            self.mask_embedding = nn.Parameter(torch.zeros(d_model))

            if orth_gain is not None:
                torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
                if value_embedding_bias:
                    self.value_embedding.bias.data.zero_()
                # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

            # Positional embedding
            if self.add_positional_embedding:
                self.position_embedding = PositionalEmbedding(d_model)

            # Residual dropout
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            """Forward Function."""
            mask = Masking.convert_seq_to_patch_view(
                mask, patch_len=self.patch_len
            ).unsqueeze(-1)
            # mask : [batch_size x n_patches x 1]
            n_channels = x.shape[1]
            mask = (
                mask.repeat_interleave(self.d_model, dim=-1)
                .unsqueeze(1)
                .repeat(1, n_channels, 1, 1)
            )
            # mask : [batch_size x n_channels x n_patches x d_model]

            # Input encoding
            x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
            if self.add_positional_embedding:
                x = x + self.position_embedding(x)

            return self.dropout(x)

    class Patching(nn.Module):
        """Patching."""

        def __init__(self, patch_len: int, stride: int):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            if self.stride != self.patch_len:
                warnings.warn(
                    "Stride and patch length are not equal. "
                    "This may lead to unexpected behavior."
                )

        def forward(self, x):
            """Forward Function."""
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # x : [batch_size x n_channels x num_patch x patch_len]
            return x
else:

    class PositionalEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class TokenEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class FixedEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class TimeFeatureEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class TemporalEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class DataEmbedding_wo_pos:
        """Dummy class if torch is unavailable."""

        pass

    class DataEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class PatchEmbedding:
        """Dummy class if torch is unavailable."""

        pass

    class Patching:
        """Dummy class if torch is unavailable."""

        pass
