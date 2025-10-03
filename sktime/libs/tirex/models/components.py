# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.


from dataclasses import dataclass, field
from typing import Any, Optional
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


SCALER_STATE = "scaler_state"


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden_layer = torch.nn.Linear(in_dim, h_dim)
        self.output_layer = torch.nn.Linear(h_dim, out_dim)
        self.residual_layer = torch.nn.Linear(in_dim, out_dim)
        self.act = torch.nn.ReLU()

    def forward(self, x: Any):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out


@dataclass
class StandardScaler:
    eps: float = 1e-5
    nan_loc: float = 0.0

    def scale(
        self,
        x: Any,
        loc_scale: tuple[Any, Any] | None = None,
    ) -> tuple[Any, tuple[Any, Any]]:
        if loc_scale is None:
            loc = torch.nan_to_num(
                torch.nanmean(x, dim=-1, keepdim=True), nan=self.nan_loc
            )
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return ((x - loc) / scale), (loc, scale)

    def re_scale(self, x: Any, loc_scale: tuple[Any, Any]) -> Any:
        loc, scale = loc_scale
        return x * scale + loc


@dataclass
class _Patcher:
    patch_size: int
    patch_stride: int
    left_pad: bool

    def __post_init__(self):
        assert self.patch_size % self.patch_stride == 0

    def __call__(self, x: Any) -> Any:
        assert x.ndim == 2
        length = x.shape[-1]

        if length < self.patch_size or (length % self.patch_stride != 0):
            if length < self.patch_size:
                padding_size = (
                    *x.shape[:-1],
                    self.patch_size - (length % self.patch_size),
                )
            else:
                padding_size = (
                    *x.shape[:-1],
                    self.patch_stride - (length % self.patch_stride),
                )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            if self.left_pad:
                x = torch.concat((padding, x), dim=-1)
            else:
                x = torch.concat((x, padding), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


@dataclass
class PatchedUniTokenizer:
    patch_size: int
    scaler: Any = field(default_factory=StandardScaler)
    patch_stride: int | None = None

    def __post_init__(self):
        if self.patch_stride is None:
            self.patch_stride = self.patch_size
        self.patcher = _Patcher(self.patch_size, self.patch_stride, left_pad=True)

    def context_input_transform(self, data: Any):
        assert data.ndim == 2
        data, scale_state = self.scaler.scale(data)
        return self.patcher(data), {SCALER_STATE: scale_state}

    def output_transform(self, data: Any, tokenizer_state: dict):
        data_shape = data.shape
        data = self.scaler.re_scale(
            data.reshape(data_shape[0], -1), tokenizer_state[SCALER_STATE]
        ).view(*data_shape)
        return data


class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger
    instance."""

    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""  # Buffer for partial lines

    def write(self, message):
        # Filter out empty messages (often from just a newline)
        if message.strip():
            self.linebuf += message
            # If the message contains a newline, process the full line
            if "\n" in self.linebuf:
                lines = self.linebuf.splitlines(keepends=True)
                for line in lines:
                    if line.endswith("\n"):
                        # Log full lines without the trailing newline (logger adds its own)
                        self.logger.log(self.log_level, line.rstrip("\n"))
                    else:
                        # Keep partial lines in buffer
                        self.linebuf = line
                        return
                self.linebuf = ""  # All lines processed
            # If no newline, keep buffering

    def flush(self):
        # Log any remaining buffered content when flush is called
        if self.linebuf.strip():
            self.logger.log(self.log_level, self.linebuf.rstrip("\n"))
            self.linebuf = ""
