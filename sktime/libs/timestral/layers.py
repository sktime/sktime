"""Transformer layers for Timestral inference.

Parameter names match the released weights, so they load with
`strict=True`. All tensors use the packed layout: one row per variate of every series in
the batch, `(R, N, D)` with `N` patches. Time attention runs per row over patches; variate
attention runs per patch over the rows of the same series.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sktime.libs.timestral.config import LAYER_KIND_AXES


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.scale


class ResidualBlock(nn.Module):
    """`output(silu(hidden(x))) + residual(x)`, used by the patch embedding and the head."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.output_layer(self.activation(self.hidden_layer(x)))
        return out + self.residual_layer(x)


class FusedSwiGLU(nn.Module):
    """SwiGLU MLP with one fused input projection; computes `gate * silu(up)`."""

    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool) -> None:
        super().__init__()
        self.w12 = nn.Linear(embed_dim, 2 * hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(gate * F.silu(up))


class SinCosRoPE(nn.Module):
    """Rotary position embedding (sin/cos concatenation), with partial rotation.

    Only the first `rotary_dim` dimensions of each head rotate. The inverse-frequency
    table is zero-padded past `rotary_dim / 2`, so the remaining dimensions get `cos=1`,
    `sin=0`, which is the identity rotation.
    """

    inv_freq: Tensor

    def __init__(self, dim: int, base: float, rotary_dim: int) -> None:
        super().__init__()
        assert 0 < rotary_dim <= dim and rotary_dim % 2 == 0, f"invalid rotary_dim={rotary_dim} for dim={dim}"
        self.dim = dim
        self.base = base
        self.rotary_dim = rotary_dim
        self.register_buffer("inv_freq", self._build_inv_freq(None), persistent=False)

    def _build_inv_freq(self, device: torch.device | None) -> Tensor:
        inv_freq = torch.zeros(self.dim // 2, dtype=torch.float32, device=device)
        idx = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
        inv_freq[: self.rotary_dim // 2] = 1.0 / (self.base ** (idx / self.rotary_dim))
        return inv_freq

    def rebuild_inv_freq(self) -> None:
        """Recompute the table on the buffer's current device.

        A table computed on CPU and then moved could differ in the last bit.
        """
        self.inv_freq = self._build_inv_freq(self.inv_freq.device)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(cos, sin)` tables, each `(R, N, head_dim)`, for integer `position_ids` `(R, N)`."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate `q`, `k` shaped `(batch, seq, heads, head_dim)`; the tables broadcast over heads."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


@dataclass(frozen=True)
class TwoStreamMask:
    """Time mask over the `[Enc ; Dec]` sequence plus its split point.

    Attention with this mask runs two query passes: context queries `[:enc_len]` over
    context keys only, and horizon queries `[enc_len:]` over all keys. This skips the
    always-empty context-to-horizon quadrant.
    """

    mask: Tensor  # (R, 1, S, S) bool, True = attend
    enc_len: int


def _sdpa(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None, is_causal: bool = False) -> Tensor:
    """Scaled dot-product attention on `(batch, seq, heads, head_dim)`; returns `(batch, seq, heads * head_dim)`."""
    batch, seqlen, _, _ = query.shape
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        is_causal=is_causal,
        dropout_p=0.0,
    )
    return out.transpose(1, 2).contiguous().view(batch, seqlen, -1)


class Attention(nn.Module):
    """Multi-head attention over the time axis (`is_time=True`) or the variate axis.

    Time attention applies RoPE; variate attention has no positional encoding. With the
    `sdpa_sigmoid` gate, the attention output is multiplied by `sigmoid(x @ W_g)` before
    the output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, is_time: bool, use_bias: bool, gate: str) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.wQKV = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.wO = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self._gate = nn.Linear(embed_dim, embed_dim, bias=use_bias) if gate == "sdpa_sigmoid" else None

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split the fused projection. The weight layout is `(qkv, head_dim, n_heads)`."""
        batch, seq, _ = x.shape
        qkv = self.wQKV(x.contiguous()).reshape(batch, seq, 3, self.head_dim, self.num_heads)
        q, k, v = qkv.permute(2, 0, 1, 4, 3).unbind(dim=0)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def forward(self, x: Tensor, mask: Tensor | TwoStreamMask | None, cos: Tensor, sin: Tensor) -> Tensor:
        rows, num_patches, _ = x.shape
        inputs = x if self.is_time else x.permute(1, 0, 2)  # time: (R, N, D); variate: (N, R, D)
        q, k, v = self._qkv(inputs)
        if self.is_time:
            q, k = _apply_rotary(q, k, cos, sin)
        if isinstance(mask, TwoStreamMask):
            assert self.is_time, "the context/horizon split applies to time attention only"
            n = mask.enc_len
            out_enc = _sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
            out_dec = _sdpa(q[:, n:], k, v, mask.mask[:, :, n:, :])
            out = torch.cat([out_enc, out_dec], dim=1)
        else:
            out = _sdpa(q, k, v, mask, is_causal=self.is_time and mask is None)
        if self._gate is not None:
            out = out * self._gate(inputs).sigmoid()
        if self.is_time:
            out = out.reshape(rows, num_patches, self.embed_dim)
        else:
            out = out.reshape(num_patches, rows, self.embed_dim).permute(1, 0, 2)
        return self.wO(out)


class TransformerBlock(nn.Module):
    """The attention(s) of one layer kind, in order, each a pre-norm residual, then one MLP."""

    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = nn.ModuleList([RMSNorm(embed_dim, norm_eps) for _ in axes])
        self.attns = nn.ModuleList(
            [
                Attention(embed_dim, num_heads, is_time, attn_bias, time_attn_gate if is_time else variate_attn_gate)
                for is_time in axes
            ]
        )
        self.mlp_norm = RMSNorm(embed_dim, norm_eps)
        self.mlp = FusedSwiGLU(embed_dim, mlp_hidden_dim, mlp_bias)

    def forward(
        self, x: Tensor, time_mask: Tensor | TwoStreamMask | None, variate_mask: Tensor | None, cos: Tensor, sin: Tensor
    ) -> Tensor:
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(norm(x), time_mask if attn.is_time else variate_mask, cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    kind=kind,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_eps=norm_eps,
                    attn_bias=attn_bias,
                    mlp_bias=mlp_bias,
                    time_attn_gate=time_attn_gate,
                    variate_attn_gate=variate_attn_gate,
                )
                for kind in block_schedule
            ]
        )
        self.rope = SinCosRoPE(dim=embed_dim // num_heads, base=rope_theta, rotary_dim=rope_dim)
        self.out_norm = RMSNorm(embed_dim, norm_eps)

    def forward(
        self, x: Tensor, time_mask: Tensor | TwoStreamMask | None, variate_mask: Tensor | None, positions: Tensor
    ) -> Tensor:
        """Run the stack on `x` `(R, N, D)` with integer time `positions` `(R, N)`."""
        assert positions.shape == x.shape[:2], f"positions {tuple(positions.shape)} do not match x {tuple(x.shape)}"
        cos, sin = self.rope(x, positions)
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.out_norm(x)
