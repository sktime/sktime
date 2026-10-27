"""Inference configuration for Timestral checkpoints.

Only the fields that change the forward pass at inference are kept. The package
implements one family of Timestral models: causal modeling trained with Decoupled Horizon
Forecasting (DHF), with monotone quantiles, decoded in the advancing refine
frame.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

LAYER_KINDS = ("time", "variate", "time_variate", "variate_time")
ATTENTION_GATES = ("none", "sdpa_sigmoid")

# Ordered attention axes each layer kind runs before its shared MLP; True = time axis.
LAYER_KIND_AXES: dict[str, tuple[bool, ...]] = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}


@dataclass(frozen=True)
class TimestralConfig:
    """Architecture and decode settings of a Timestral checkpoint.

    Attributes:
        patch_size: Time steps per patch.
        d_model: Model width.
        num_heads: Attention heads per attention sublayer.
        mlp_hidden_dim: Hidden width of the SwiGLU MLP.
        block_schedule: One entry per layer, each one of `LAYER_KINDS`.
        quantile_knots: Quantile levels of the head, symmetric about 0.5.
        block_size: Trained DHF block size K, the maximum patches decoded per block.
        context_length: Trained context length, the default context cap.
        quantile_width_factor: Increment scale of the monotone quantile head.
    """

    patch_size: int
    d_model: int
    num_heads: int
    mlp_hidden_dim: int
    block_schedule: tuple[str, ...]
    quantile_knots: tuple[float, ...]
    block_size: int
    context_length: int
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-6
    attn_bias: bool = False
    mlp_bias: bool = False
    time_attn_gate: str = "sdpa_sigmoid"
    variate_attn_gate: str = "sdpa_sigmoid"
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    stabilize_with_global: bool = False
    covariate_enabled: bool = True
    split_context_horizon_attn: bool = True
    attend_missing_patches: bool = True
    quantile_width_factor: float = 2.0

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, f"d_model={self.d_model} not divisible by num_heads={self.num_heads}"
        assert self.rope_dim > 0 and self.rope_dim % 2 == 0, (
            f"rope_partial_factor={self.rope_partial_factor} gives an invalid rope_dim={self.rope_dim}"
        )
        for kind in self.block_schedule:
            assert kind in LAYER_KINDS, f"unknown layer kind {kind!r}; expected one of {LAYER_KINDS}"
        for gate in (self.time_attn_gate, self.variate_attn_gate):
            assert gate in ATTENTION_GATES, f"unknown attention gate {gate!r}; expected one of {ATTENTION_GATES}"
        knots = self.quantile_knots
        assert 0.5 in knots, f"quantile_knots must contain 0.5, got {knots}"
        assert all(a < b for a, b in zip(knots, knots[1:])), f"quantile_knots must be increasing, got {knots}"
        if self.stabilize_with_global:
            raise ValueError("the advancing refine frame needs stabilize_with_global=False")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        """Head dimensions rotated by RoPE; the rest pass through unrotated."""
        return int(self.head_dim * self.rope_partial_factor)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @property
    def median_index(self) -> int:
        return self.quantile_knots.index(0.5)

    @property
    def supports_known_covariates(self) -> bool:
        """Whether the layer schedule can route a known covariate's future to the targets.

        Some time op must be followed by at least one more time op and at least one
        variate op; otherwise the future of a known covariate never reaches a target.
        """
        if not self.covariate_enabled:
            return False
        ops = [is_time for kind in self.block_schedule for is_time in LAYER_KIND_AXES[kind]]
        for i, is_time in enumerate(ops):
            rest = ops[i + 1 :]
            if is_time and any(rest) and not all(rest):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["block_schedule"] = list(self.block_schedule)
        d["quantile_knots"] = list(self.quantile_knots)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimestralConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d = dict(d)
        d["block_schedule"] = tuple(d["block_schedule"])
        d["quantile_knots"] = tuple(float(q) for q in d["quantile_knots"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TimestralConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
