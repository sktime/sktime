# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

"""Patch transformer producing nine quantiles of the next `prediction_length` steps."""

import json
import math
from contextlib import nullcontext
from pathlib import Path

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")

from .config import SimonsConfig
from .covariates import (
    ROLE_BINARY,
    ROLE_TARGET,
    CovariateModules,
    covariate_rows,
    read_mask,
)
from .periodicity import detect_periods, period_rotary, select_periods

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SCALE = 1e-5
MAX_VARIATES = 32


def sliding_scale(values: torch.Tensor, window: int):
    """Causal (loc, std) at every step of [..., T] values over the observed values among the last
    `window` steps, frozen across missing steps (loc 0, std MIN_SCALE before the first one)."""
    observed = torch.isfinite(values)
    ref = torch.nanmedian(values.float(), dim=-1, keepdim=True).values.double()
    ref = torch.where(torch.isfinite(ref), ref, 0)
    x = torch.where(observed, values.double() - ref, 0)
    w = observed.double()

    def windowed(a):
        c = a.cumsum(-1)
        if window >= a.shape[-1]:
            return c
        return c - F.pad(c[..., :-window], (window, 0))

    n = windowed(w)
    s1 = windowed(x)
    s2 = windowed(x * x)
    mean = s1 / n.clamp_min(1)
    var = ((s2 - s1 * mean) / (n - 1).clamp_min(1)).clamp_min(MIN_SCALE**2)
    loc, std = mean + ref, var.sqrt()
    position = torch.arange(values.shape[-1], device=values.device)
    last = torch.where(observed, position, -1).cummax(-1).values
    loc = torch.where(last >= 0, loc.gather(-1, last.clamp_min(0)), 0)
    std = torch.where(last >= 0, std.gather(-1, last.clamp_min(0)), MIN_SCALE)
    return loc, std


def scale_values(values: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
    """asinh of the scaled values (0 where missing) and the observed flags."""
    observed = torch.isfinite(values)
    clean = torch.where(observed, values, 0).double()
    scaled = (clean - loc) / scale
    return torch.where(observed, scaled, 0).asinh().float(), observed


def tokenize(scaled: torch.Tensor, observed: torch.Tensor, patch: int) -> torch.Tensor:
    """[..., T] scaled values and flags -> [..., T / patch, 2 * patch] tokens."""
    return torch.cat(
        [
            scaled.unflatten(-1, (-1, patch)),
            observed.float().unflatten(-1, (-1, patch)),
        ],
        -1,
    )


def future_steps(config) -> int:
    """Steps covered by the future patches of one pass."""
    patch = config.patch_length
    return math.ceil(config.prediction_length / patch) * patch


def layout(config, length: int):
    """(far_len raw steps, far tokens) for a [.., length] sequence (history + future steps), or
    None when its history fits in the near region (no far tokens)."""
    patch = config.patch_length
    history = length - future_steps(config)
    if history <= config.near_context:
        return None
    far_len = history - config.near_context
    block = patch * config.far_pool
    if far_len % block:
        raise ValueError(
            f"far region {far_len} must be a multiple of patch * far_pool {block}"
        )
    return far_len, far_len // block


def front(
    model,
    values: torch.Tensor,
    far_on: bool = True,
    fixed: torch.Tensor | None = None,
    skip: int = 0,
    periods=None,
    return_periods: bool = False,
):
    """[B,V,T] values, fixed [B,V] -> tokens [B,V,S,2p], rotary [B*V,1,S,half],
    started/loc/scale [B,V,S], far [S]/None, far count; optional periods [B*V,slots].
    `skip` omits an all-missing far prefix, keeping full-window positions and period windows."""
    c = model.config
    patch = c.patch_length
    batch, variates, length = values.shape
    history = length + skip - future_steps(c)  # of the full window
    half = c.width // c.heads // 2
    spec = layout(c, length + skip)
    if skip and (
        spec is None or skip % (patch * c.far_pool) or skip > spec[0] or not far_on
    ):
        raise ValueError(
            "skip must be a multiple of the far block within the far region"
        )
    if periods is None:
        if skip:
            raise ValueError("skip needs a periods function")
        periods = lambda v, h, f, _: select_periods(c, v, h, f)  # noqa: E731
    loc_all, std_all = sliding_scale(values, c.stats_window)
    if fixed is not None:
        loc_all = torch.where(fixed[..., None], 0.0, loc_all)
        std_all = torch.where(fixed[..., None], 1.0, std_all)
    device = values.device
    if spec is None:
        patches = length // patch
        end = slice(patch - 1, None, patch)
        loc, scale = loc_all[..., end], std_all[..., end]
        scaled, observed = scale_values(
            values, loc.repeat_interleave(patch, -1), scale.repeat_interleave(patch, -1)
        )
        tokens = tokenize(scaled, observed, patch)
        starts = torch.arange(patches, device=device, dtype=torch.float64) * patch
        found = periods(values, history, 0, 0).reshape(batch * variates, -1)
        rotary = period_rotary(found, starts, patch, half)
        started = (
            observed.unflatten(-1, (patches, patch))
            .any(-1)
            .int()
            .cummax(-1)
            .values.bool()
        )
        out = (tokens, rotary, started, loc, scale, None, 0)
        return (*out, found) if return_periods else out

    pool = c.far_pool
    block = patch * pool
    far_len, far_tokens = spec[0] - skip, spec[1] - skip // block
    near = values[..., far_len:]
    near_patches = near.shape[-1] // patch
    end = slice(far_len + patch - 1, None, patch)
    near_loc, near_scale = loc_all[..., end], std_all[..., end]
    scaled, near_seen = scale_values(
        near,
        near_loc.repeat_interleave(patch, -1),
        near_scale.repeat_interleave(patch, -1),
    )
    near_tok = tokenize(scaled, near_seen, patch)
    raw = values[..., :far_len].unflatten(-1, (far_len // pool, pool))
    seen = torch.isfinite(raw)
    count = seen.sum(-1)
    total = torch.where(seen, raw, 0).double().sum(-1)
    pooled = torch.where(count > 0, total / count.clamp_min(1), float("nan"))
    end = slice(block - 1, far_len, block)
    far_loc, far_scale = loc_all[..., end], std_all[..., end]
    scaled, far_seen = scale_values(
        pooled,
        far_loc.repeat_interleave(patch, -1),
        far_scale.repeat_interleave(patch, -1),
    )
    if c.far_min_history < history:
        old = torch.isfinite(
            values[..., : max(0, history - c.far_min_history - skip)]
        ).any(-1)
        far_seen = far_seen & old[..., None]
        scaled = torch.where(far_seen, scaled, 0)
    else:
        far_seen = torch.zeros_like(far_seen)
        scaled = torch.zeros_like(scaled)
    far_tok = tokenize(scaled, far_seen, patch)
    tokens = torch.cat([far_tok, near_tok], 2)
    loc = torch.cat([far_loc, near_loc], -1)
    scale = torch.cat([far_scale, near_scale], -1)
    tokens_seen = torch.cat(
        [
            far_seen.unflatten(-1, (far_tokens, patch)).any(-1),
            near_seen.unflatten(-1, (near_patches, patch)).any(-1),
        ],
        -1,
    )
    started = tokens_seen.int().cummax(-1).values.bool()
    starts = skip + torch.cat(
        [
            torch.arange(far_tokens, device=device, dtype=torch.float64) * block,
            far_len
            + torch.arange(near_patches, device=device, dtype=torch.float64) * patch,
        ]
    )
    found = periods(values, history - skip, far_len, skip).reshape(batch * variates, -1)
    rotary = period_rotary(found, starts, patch, half)
    far = torch.arange(far_tokens + near_patches, device=device) < far_tokens
    if not far_on:
        f = far_tokens
        rotary = tuple(t[:, :, f:] for t in rotary)
        return (
            tokens[:, :, f:],
            rotary,
            started[..., f:],
            loc[..., f:],
            scale[..., f:],
            far[f:],
            0,
        )
    out = (tokens, rotary, started, loc, scale, far, far_tokens)
    return (*out, found) if return_periods else out


def sign_bounds(
    values: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    patch: int,
    far_len: int,
):
    """Return per-token (lo, hi) [B,V,S] in asinh units from one-signed observed prefixes.
    Far tokens remain unbounded."""
    near = values[..., far_len:]
    seen = torch.isfinite(near).unflatten(-1, (-1, patch))
    blocks = near.unflatten(-1, (-1, patch))
    low = torch.where(seen, blocks, math.inf).amin(-1)
    high = torch.where(seen, blocks, -math.inf).amax(-1)
    far = (
        values[..., :far_len]
        if far_len
        else values[..., :1].new_full((*values.shape[:-1], 1), float("nan"))
    )
    far_seen = torch.isfinite(far)
    low = torch.cat(
        [torch.where(far_seen, far, math.inf).amin(-1, keepdim=True), low], -1
    )
    high = torch.cat(
        [torch.where(far_seen, far, -math.inf).amax(-1, keepdim=True), high], -1
    )
    low, high = low.cummin(-1).values[..., 1:], high.cummax(-1).values[..., 1:]
    count, tokens = low.shape[-1], loc.shape[-1]
    edge = torch.asinh(
        -loc[..., tokens - count :].double() / scale[..., tokens - count :].double()
    )
    any_seen = torch.isfinite(low)
    nonneg = any_seen & (low >= 0)
    nonpos = any_seen & (high <= 0) & (low < 0)
    lo = torch.where(nonneg, edge.float(), -math.inf)
    hi = torch.where(nonpos, edge.float(), math.inf)
    lo = F.pad(lo, (tokens - count, 0), value=-math.inf)
    hi = F.pad(hi, (tokens - count, 0), value=math.inf)
    return lo, hi


def rotate(x: torch.Tensor, rotary) -> torch.Tensor:
    cos, sin = (t.to(x.dtype) for t in rotary)
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


CROP_STEP = 8
SKIP_TOKENS = (
    CROP_STEP  # forward skips leading missing far tokens in multiples of this many
)


def row_crops(
    start: torch.Tensor, roles: torch.Tensor, batch: int, variates: int
) -> torch.Tensor:
    """Return [B*V] first required token columns, rounded down to CROP_STEP.
    Each row must retain its own and its targets' observed prefixes."""
    size = start.shape[-1]
    first = torch.where(start.any(-1), start.int().argmax(-1), size).reshape(
        batch, variates
    )
    target = torch.where(roles == ROLE_TARGET, first, size).amin(-1, keepdim=True)
    target = torch.where(target < size, target, 0)
    return (torch.minimum(first, target).reshape(-1) // CROP_STEP) * CROP_STEP


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(
        {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()
        ]
    )


def shared_rows(
    tokens: torch.Tensor,
    rotary,
    roles: torch.Tensor,
    crops: torch.Tensor,
    independent: torch.Tensor,
):
    """Return (compute [M], source [N]) so row n uses compute[source[n]]; (None,None) if unique.
    Reuse requires equal tokens [N,S,F], rotary tables, roles and crops."""
    n, device = tokens.shape[0], tokens.device
    index = torch.arange(n, device=device)
    parts = [tokens.reshape(n, -1), *(t.reshape(n, -1) for t in rotary)]
    generator = torch.Generator().manual_seed(0)
    keys = [
        torch.stack(
            [
                roles.double(),
                crops.double(),
                torch.where(independent, -1, index).double(),
            ],
            1,
        )
    ]
    for p in parts:
        weights = torch.rand(p.shape[1], 2, generator=generator, dtype=torch.float64)
        keys.append(p.double() @ weights.to(device))
    _, group = torch.unique(torch.cat(keys, 1), dim=0, return_inverse=True)
    first = torch.full((int(group.max()) + 1,), n, device=device)
    rep = first.scatter_reduce(0, group, index, "amin")[group]
    same = (roles == roles[rep]) & (crops == crops[rep])
    for p in parts:
        bits = _bits(p)
        same &= (bits == bits[rep]).all(-1)
    rep = torch.where(same, rep, index)
    compute = torch.unique(rep)
    if len(compute) == n:
        return None, None
    return compute, torch.searchsorted(compute, rep)


def crop_groups(crops: torch.Tensor) -> list:
    """Return (crop, rows [R]) for each distinct value of crops [N]."""
    return [
        (crop, torch.nonzero(crops == crop)[:, 0])
        for crop in torch.unique(crops).tolist()
    ]


def from_position(x: list, groups: list, before: torch.Tensor, first: int, size: int):
    """Return [N,S-first,W] from the groups' activations x (each [R,S-crop,W], from its crop).
    Positions before a row's crop hold its `before` [N,W] vector."""
    out = (
        before.new_zeros(len(before), size - first, before.shape[-1]) + before[:, None]
    )
    for (crop, rows), xs in zip(groups, x):
        out[rows, max(crop - first, 0) :] = xs[:, max(first - crop, 0) :]
    return out


def time_blocks(blocks, x, start: torch.Tensor, rotary, groups=None):
    """Run time blocks on x [N,S,W] with started flags [N,S]. With `groups` (see `crop_groups`),
    x and the result list each group's activations [R,S-crop,W] from its crop."""

    def run(x, start, rotary):
        position = torch.arange(x.shape[1], device=x.device)
        query, key = position[:, None], position[None, :]
        mask = ((key <= query)[None] & (start[:, None, :] | (key == query)[None]))[
            :, None
        ]
        for block in blocks:
            x = block(x, mask, rotary=rotary)
        return x

    if groups is None:
        return run(x, start, rotary)
    out = []
    for (crop, rows), xs in zip(groups, x):
        ss = start[rows, crop:]
        rs = tuple(t[rows][:, :, crop:] for t in rotary)
        shared, length = constant_prefix(xs, ss)
        if shared is None:
            out.append(run(xs, ss, rs))
            continue
        rest = torch.ones(len(rows), dtype=torch.bool, device=xs.device)
        rest[shared] = False
        y = torch.empty_like(xs)
        y[shared] = prefix_blocks(
            blocks, xs[shared], tuple(t[shared] for t in rs), length
        )
        if bool(rest.any()):
            y[rest] = run(xs[rest], ss[rest], tuple(t[rest] for t in rs))
        out.append(y)
    return out


PREFIX_MIN = 8  # shortest shared prefix (positions) that `time_blocks` runs once


def constant_prefix(x: torch.Tensor, start: torch.Tensor):
    """Return (rows [R], P) sharing one started prefix vector in x [N,S,W], start [N,S].
    PREFIX_MIN <= P < S; return (None, 0) if no rows qualify."""
    size = x.shape[1]
    if size <= PREFIX_MIN:
        return None, 0
    same = (x == x[:, :1]).all(-1) & start
    run = same.int().cumprod(-1).sum(-1)
    candidate = run >= PREFIX_MIN
    if not bool(candidate.any()):
        return None, 0
    first = int(torch.nonzero(candidate)[0, 0])
    candidate &= (x[:, 0] == x[first, 0]).all(-1)
    rows = torch.nonzero(candidate)[:, 0]
    return rows, min(int(run[rows].min()), size - 1)


def prefix_blocks(blocks, x: torch.Tensor, rotary, length: int) -> torch.Tensor:
    """Run causal blocks on x [R,S,W] with a shared, started prefix of `length` positions.
    Prefix attention returns the shared value regardless of weights; compute that state once."""
    n, size, width = x.shape
    prefix, x = x[0, 0], x[:, length:]
    later = size - length
    cos, sin = rotary
    head_rotary = (cos[:, :, :length], sin[:, :, :length])
    tail_rotary = (cos[:, :, length:], sin[:, :, length:])
    query = torch.arange(later, device=x.device)[:, None] + length
    mask = (torch.arange(size, device=x.device)[None] <= query)[None, None]
    for block in blocks:
        heads = block.heads
        qkv = block.qkv(block.attention_norm(prefix[None, None])).view(
            1, 1, 3, heads, -1
        )
        key, value = (t.transpose(1, 2) for t in qkv.unbind(2)[1:])  # [1,H,1,D]
        q, k, v = (
            block.qkv(block.attention_norm(x)).view(n, later, 3, heads, -1).unbind(2)
        )
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = rotate(q, tail_rotary), rotate(k, tail_rotary)
        keys = torch.cat([rotate(key.expand(n, heads, length, -1), head_rotary), k], 2)
        values = torch.cat([value.expand(n, heads, length, -1), v], 2)
        attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + block.out(attended.transpose(1, 2).reshape(n, later, width))
        x = x + block.mlp(block.mlp_norm(x))
        prefix = prefix + block.out(value.transpose(1, 2).reshape(1, 1, width))[0, 0]
        prefix = prefix + block.mlp(block.mlp_norm(prefix[None, None]))[0, 0]
    return torch.cat([prefix.expand(n, length, width).to(x.dtype), x], 1)


def attend_targets(
    block,
    x,
    source,
    mine,
    mask,
    batch: int,
    variates: int,
    targets: int,
    size: int,
    explicit: bool = False,
    first: int = 0,
) -> torch.Tensor:
    """Apply a cross-row block to each item's first T rows: x [M,S-first,W] -> [B*T,S-first,W].
    x holds positions from `first`; `source` [B*V] and `mine` [B*T] index it; mask [B*S,1,T,V]
    selects keys from all rows. `explicit` uses fp32 softmax."""
    if first:
        mask = mask.view(batch, size, *mask.shape[1:])[:, first:].flatten(0, 1)
        size -= first
    width, heads = x.shape[-1], block.heads
    weight = block.qkv.weight
    kv = F.linear(block.attention_norm(x), weight[width:])[source]  # [N,S,2W]
    kv = kv.view(batch, variates, size, 2, heads, -1).permute(3, 0, 2, 4, 1, 5)
    k, v = (t.reshape(batch * size, heads, variates, -1) for t in kv.unbind(0))
    current = x[mine]  # [B*T,S,W]
    q = F.linear(block.attention_norm(current), weight[:width])
    q = (
        q.view(batch, targets, size, heads, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * size, heads, targets, -1)
    )
    if explicit:
        scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        weights = scores.float().masked_fill(~mask, float("-inf")).softmax(-1)
        attended = weights.to(v.dtype) @ v
    else:
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attended = (
        attended.view(batch, size, heads, targets, -1)
        .permute(0, 3, 1, 2, 4)
        .reshape(batch * targets, size, width)
    )
    y = current + block.out(attended)
    return y + block.mlp(block.mlp_norm(y))


class Block(nn.Module):
    """Pre-norm attention + GELU MLP block; causal blocks apply the rotary embedding."""

    def __init__(self, config: SimonsConfig, causal: bool):
        super().__init__()
        self.heads, self.causal = config.heads, causal
        self.attention_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.mlp_norm = nn.RMSNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, rotary=None) -> torch.Tensor:
        n, s, width = x.shape
        q, k, v = (
            self.qkv(self.attention_norm(x)).view(n, s, 3, self.heads, -1).unbind(2)
        )
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.causal:
            q, k = rotate(q, rotary), rotate(k, rotary)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(attended.transpose(1, 2).reshape(n, s, width))
        return x + self.mlp(self.mlp_norm(x))


class Simons(nn.Module):
    """Patch transformer for quantiles 0.1, ..., 0.9 of the next `prediction_length` steps."""

    def __init__(self, config: SimonsConfig):
        super().__init__()
        self.config = config
        patch, width = config.patch_length, config.width
        self.embed = nn.Linear(2 * patch, width)
        self.embed_mlp = nn.Sequential(
            nn.Linear(2 * patch, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.far_type = nn.Parameter(torch.zeros(width))
        self.time_blocks = nn.ModuleList(
            Block(config, causal=True) for _ in range(config.layers)
        )
        self.variate_block = Block(config, causal=False)
        self.norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, patch * len(QUANTILES))
        self.covariates = None
        if config.covariates:
            # A private stream keeps covariate initialization independent of backbone draws.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(32)
                self.covariates = CovariateModules(config, Block)

    @classmethod
    def from_pretrained(cls, name_or_path, device="cpu", revision=None) -> "Simons":
        """Load config.json and model.safetensors from a local folder or Hub repo, in eval mode.
        `revision` selects a Hub branch, tag or commit; `device` sets the model device."""
        load_file = _safe_import("safetensors.torch.load_file")

        path = Path(name_or_path).expanduser()
        if not path.is_dir():
            if (
                path.exists()
                or path.is_absolute()
                or str(name_or_path).startswith((".", "~"))
            ):
                raise FileNotFoundError(
                    f"{name_or_path} is not a model folder (config.json and "
                    "model.safetensors)"
                )
            snapshot_download = _safe_import("huggingface_hub.snapshot_download")

            path = Path(
                snapshot_download(
                    str(name_or_path),
                    revision=revision,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        model = cls(SimonsConfig.from_json(path / "config.json"))
        model.load_state_dict(load_file(path / "model.safetensors"))
        return model.to(device).eval()

    def save_pretrained(self, path):
        save_file = _safe_import("safetensors.torch.save_file")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, path / "model.safetensors")

    def forecast(
        self, history, horizon: int, series_ids=None, covariates=None, **options
    ) -> torch.Tensor:
        """Quantile forecasts; see `simons.forecast` (the function)."""
        from .forecast import forecast

        return forecast(
            self, history, horizon, series_ids, covariates=covariates, **options
        )

    def forward(
        self,
        history: torch.Tensor,
        series_ids: torch.Tensor,
        mixed: bool = False,
        covariates: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """History [B,V,context_length] -> quantiles [B,V,prediction_length,9]; NaN = missing.
        Equal series_ids [B,V] group variates; covariates [B,C,context_length+future patches].
        `steps` limits output to the patches covering it; `mixed` enables a bf16 CUDA trunk."""
        c = self.config
        patch = c.patch_length
        if history.shape[-1] != c.context_length:
            raise ValueError(
                f"history must have context_length = {c.context_length} steps"
            )
        future = math.ceil(c.prediction_length / patch)
        need = (
            future if steps is None else max(1, min(future, math.ceil(steps / patch)))
        )
        values = F.pad(history, (0, future * patch), value=float("nan"))
        skip = self._skip(values, covariates)
        if skip:
            values = values[..., skip:]
            covariates = None if covariates is None else covariates[..., skip:]
        variates, roles, fixed, head = history.shape[1], None, None, None
        if covariates is not None:
            if self.covariates is None:
                raise ValueError("this checkpoint has no covariate weights")
            grown = covariate_rows(c, values, series_ids, covariates, skip=skip)
            if grown is not None:
                values, series_ids, roles = grown[:3]
                head = grown[3] if skip else None
                fixed = (roles == ROLE_BINARY) if c.binary_rows else None
        with torch.autocast("cuda", torch.bfloat16, enabled=mixed):
            parts = front(
                self,
                values,
                fixed=fixed,
                skip=skip,
                periods=self._periods,
                return_periods=True,
            )
            tokens, rotary, started, loc, scale, _, far_tokens, _ = parts
            far_len = c.context_length - c.near_context - skip
            lo, hi = sign_bounds(values, loc, scale, patch, far_len)
            if head is not None and bool(torch.isfinite(head).any()):
                # Fit rows have an observed intercept over the skipped prefix.
                tokens, rotary, started, loc, scale, far_tokens, lo, hi = self._unskip(
                    parts, lo, hi, values[:, -1], head, skip
                )
            size = tokens.shape[2]
            last = size - future - 1
            if need < future:
                size -= future - need
                tokens, started, lo, hi = (
                    t[:, :, :size] for t in (tokens, started, lo, hi)
                )
                rotary = tuple(t[:, :, :size] for t in rotary)
            # Attention isolates unused prefix tokens; keep at least the last history token.
            any_started = started.flatten(0, 1).any(0)
            crop = int(
                torch.where(any_started, torch.arange(size, device=values.device), size)
                .amin()
                .clamp_max(last)
            )
            # The last history token predicts the first future patch.
            features = self._trunk(
                tokens[:, :, crop:],
                tuple(t[:, :, crop:] for t in rotary),
                started[:, :, crop:],
                series_ids,
                far_tokens - crop,
                mixed,
                roles,
                variates,
                size - last,
            )
            window = slice(last, last + need)
            quantiles = self._head(
                features[:, :, :need],
                lo[:, :variates, window],
                hi[:, :variates, window],
                mixed,
            )
        location = loc[:, :variates, window, None, None]
        spread = scale[:, :variates, window, None, None]
        out = quantiles.double().sinh() * spread + location
        return out.flatten(2, 3)[:, :, : c.prediction_length].float()

    def _skip(self, values: torch.Tensor, covariates: torch.Tensor | None) -> int:
        """Return unobserved far-prefix steps in multiples of SKIP_TOKENS far tokens.
        Known-future covariates occupy positions one patch earlier."""
        c = self.config
        patch = c.patch_length
        unit = SKIP_TOKENS * patch * c.far_pool
        far_len = c.context_length - c.near_context
        if far_len < unit:
            return 0
        seen = torch.isfinite(values[..., :far_len]).flatten(0, -2).any(0)
        if covariates is not None:
            cov = (
                torch.isfinite(covariates[..., : far_len + patch]).flatten(0, -2).any(0)
            )
            seen |= cov[:far_len] | cov[patch : far_len + patch]
        first = int(
            torch.where(
                seen, torch.arange(far_len, device=values.device), far_len
            ).amin()
        )
        return first // unit * unit

    def _unskip(self, parts, lo, hi, fit_values, head, skip: int):
        """Restore skipped tokens and the full fit row from head [B,skip] and fit_values [B,L].
        The fit row is last; other rows have missing prefixes."""
        c = self.config
        tokens, _, started, loc, scale, _, far_tokens, periods = parts
        patch, block = c.patch_length, c.patch_length * c.far_pool
        extra = skip // block
        tokens = F.pad(tokens, (0, 0, extra, 0))
        started = F.pad(started, (extra, 0), value=False)
        loc = F.pad(loc, (extra, 0), value=0.0)
        scale = F.pad(scale, (extra, 0), value=MIN_SCALE)
        lo = F.pad(lo, (extra, 0), value=-math.inf)
        hi = F.pad(hi, (extra, 0), value=math.inf)
        far_tokens += extra
        far_len = c.context_length - c.near_context
        device = tokens.device
        starts = torch.cat(
            [
                torch.arange(far_tokens, device=device, dtype=torch.float64) * block,
                far_len
                + torch.arange(
                    tokens.shape[2] - far_tokens, device=device, dtype=torch.float64
                )
                * patch,
            ]
        )
        cos, sin = period_rotary(periods, starts, patch, c.width // c.heads // 2)
        batch, variates = tokens.shape[:2]
        fit = front(
            self, torch.cat([head, fit_values], -1)[:, None], periods=self._periods
        )
        tokens[:, -1:] = fit[0]
        started[:, -1:] = fit[2]
        loc[:, -1:], scale[:, -1:] = fit[3], fit[4]
        cos, sin = (t.view(batch, variates, *t.shape[1:]).clone() for t in (cos, sin))
        cos[:, -1], sin[:, -1] = (t.view(batch, *t.shape[1:]) for t in fit[1])
        rotary = tuple(t.view(batch * variates, *t.shape[2:]) for t in (cos, sin))
        return tokens, rotary, started, loc, scale, far_tokens, lo, hi

    def _periods(
        self, values: torch.Tensor, history: int, far_len: int, skip: int = 0
    ) -> torch.Tensor:
        """Return periods [B,V,slots] for values [B,V,L], restoring `skip` missing steps as needed.
        The detector is sign-invariant, so equal or negated adjacent histories share periods."""
        c = self.config

        def detect(x, context):
            return detect_periods(
                x,
                context,
                c.period_slots,
                c.period_alpha,
                c.period_local,
                c.patch_length,
            )

        batch, variates, _ = values.shape
        seen = values[..., :history]
        reuse = torch.zeros(batch, variates, dtype=torch.bool, device=values.device)
        if batch > 1:
            cur, prev = seen[1:], seen[:-1]
            both = torch.isnan(cur) & torch.isnan(prev)
            reuse[1:] = ((cur == prev) | both).all(-1) | ((cur == -prev) | both).all(-1)
        rows = torch.arange(batch, device=values.device)[:, None].expand(
            batch, variates
        )
        source = torch.where(reuse, -1, rows).cummax(0).values  # first row of each run
        source = (
            source * variates + torch.arange(variates, device=values.device)
        ).reshape(-1)
        own = torch.nonzero(~reuse.reshape(-1))[:, 0]
        x = seen.reshape(batch * variates, history)[own]
        if not far_len and not skip:
            periods = detect(x, history)
        else:
            periods = detect(x[:, far_len:], history - far_len)
            full = history + skip
            mid = min(c.detector_mid, full)
            mid_start = history - mid  # < 0 after a skip into the mid window
            if mid > history - far_len:
                pick = torch.nonzero(torch.isfinite(x[:, :far_len]).any(-1))[:, 0]
                if len(pick):
                    window = F.pad(
                        x[pick, max(0, mid_start) :],
                        (max(0, -mid_start), 0),
                        value=float("nan"),
                    )
                    periods[pick] = detect(window, mid)
            if mid_start > 0:
                pick = torch.nonzero(torch.isfinite(x[:, :mid_start]).any(-1))[:, 0]
                if len(pick):
                    periods[pick] = detect(
                        F.pad(x[pick], (skip, 0), value=float("nan")), full
                    )
        out = periods.new_zeros(batch * variates, periods.shape[-1])
        out[own] = periods
        return out[source].reshape(batch, variates, -1)

    def _embed(self, tokens, far_tokens: int, mixed: bool, position=None):
        """[..., S, 2p] tokens -> [..., S, width] embeddings (fp32 under `mixed`), far type
        added; with `position` (the tokens' positions), [K, 2p] tokens -> [K, width]."""
        with torch.autocast("cuda", enabled=False) if mixed else nullcontext():
            inputs = tokens.float() if mixed else tokens
            x = self.embed(inputs) + self.embed_mlp(inputs)
        if position is None:
            position = torch.arange(tokens.shape[-2], device=tokens.device)
        far = position < far_tokens
        return x + far[..., None].to(x.dtype) * self.far_type

    def _trunk(
        self,
        tokens,
        rotary,
        started,
        series_ids,
        far_tokens: int,
        mixed: bool,
        roles: torch.Tensor | None = None,
        targets: int | None = None,
        tail: int | None = None,
    ):
        """Return [B,T,tail,width] features of the first T = `targets` rows (default: all) at the
        last `tail` positions (default: all); the first `far_tokens` tokens are pooled history.
        `roles` [B,V]: covariate rows' roles (covariate rows follow the targets); the tail must
        then start at or after every target row's first observed token."""
        batch, variates, size, _ = tokens.shape
        tail = size if tail is None else tail
        start = started.reshape(batch * variates, size)
        ids = series_ids[:, None, :].expand(batch, size, variates)
        related = (ids[..., :, None] == ids[..., None, :]).reshape(
            batch * size, 1, variates, variates
        )
        blocks = list(self.time_blocks)
        if roles is None:
            x = self._embed(tokens, far_tokens, mixed).reshape(
                batch * variates, size, -1
            )
            x = time_blocks(blocks, x, start, rotary, None)
            x = x.reshape(batch, variates, size, -1)[:, :, size - tail :]
            related = related.view(batch, size, 1, variates, variates)[:, size - tail :]
            x = x.transpose(1, 2).reshape(batch * tail, variates, -1)
            x = self.variate_block(x, related.flatten(0, 1))
            return x.reshape(batch, tail, variates, -1).transpose(1, 2)
        # Equal rows can share computation only while they read no other row.
        targets = variates if targets is None else targets
        read_after = self.config.read_after if self.covariates.read is not None else -1
        crops = row_crops(start, roles, batch, variates)
        reads = torch.zeros(batch * variates, dtype=torch.bool, device=tokens.device)
        if read_after >= 0:
            _, update = read_mask(
                roles, related.reshape(batch, size, 1, variates, variates)[:, 0], 1
            )
            reads = update.reshape(-1)
        compute, source = shared_rows(
            tokens.reshape(batch * variates, size, -1),
            rotary,
            roles.reshape(-1),
            crops,
            ~reads,
        )
        if compute is None:
            compute = source = torch.arange(batch * variates, device=tokens.device)
        # Time blocks ignore positions before each row's crop, so x keeps each crop's rows from
        # it: a whole-window fit row does not make the short rows of its item any longer.
        own = tokens.reshape(batch * variates, size, -1)[compute]
        later = (
            torch.arange(size, device=tokens.device)[None] >= crops[compute][:, None]
        )
        embedded = self._embed(
            own[later], far_tokens, mixed, torch.nonzero(later)[:, 1]
        )
        role = (
            self.covariates.role_vectors(roles)
            .reshape(batch * variates, -1)[compute]
            .to(embedded.dtype)
        )
        index = torch.zeros_like(later, dtype=torch.long)
        index[later] = torch.arange(len(embedded), device=tokens.device)
        groups = crop_groups(crops[compute])
        x = [embedded[index[r, crop:]] + role[r][:, None] for crop, r in groups]
        part = (start[compute], tuple(t[compute] for t in rotary), groups)
        cut = read_after + 1 if read_after >= 0 else len(blocks)
        x = time_blocks(blocks[:cut], x, *part)
        rows = (
            torch.arange(batch, device=tokens.device)[:, None] * variates
            + torch.arange(targets, device=tokens.device)
        ).reshape(-1)  # target rows [B*T]
        if read_after >= 0:
            mask, update = read_mask(roles, related, size)
            mine = source[rows]
            first = int(crops[rows].min())
            seen = from_position(x, groups, role, first, size)
            y = attend_targets(
                self.covariates.read,
                seen,
                source,
                mine,
                mask[:, :, :targets],
                batch,
                variates,
                targets,
                size,
                explicit=True,
                first=first,
            )
            u = (
                update[:, :targets]
                .reshape(batch, size, targets, 1)[:, first:]
                .transpose(1, 2)
                .reshape(batch * targets, size - first, 1)
                .to(y.dtype)
            )
            current = seen[mine]
            y = current + u * (y - current)
            keep = reads[rows]  # Reading targets cannot share rows.
            seen[mine[keep]] = y[keep]
            # Target rows' crops are at or after `first`, and nothing reads a row before its crop.
            x = [
                xs if crop < first else seen[r, crop - first :]
                for (crop, r), xs in zip(groups, x)
            ]
            x = time_blocks(blocks[cut:], x, *part)
        last = from_position(x, groups, role, size - tail, size)
        y = attend_targets(
            self.variate_block,
            last,
            source,
            source[rows],
            related[:, :, :targets],
            batch,
            variates,
            targets,
            size,
            first=size - tail,
        )
        return y.reshape(batch, targets, tail, -1)

    def _head(self, x, lo, hi, mixed: bool) -> torch.Tensor:
        """Map [...,width] to [...,patch,9] sorted asinh quantiles with optional sign bounds."""
        with torch.autocast("cuda", enabled=False) if mixed else nullcontext():
            h = self.norm(x.float() if mixed else x)
            raw = self.head(h.float() if mixed else h)
        q = (
            raw.float()
            .unflatten(-1, (self.config.patch_length, len(QUANTILES)))
            .sort(-1)
            .values
        )
        if not self.config.sign_clip:
            return q
        lo = torch.nan_to_num(lo.float(), nan=-math.inf)[..., None, None]
        hi = torch.nan_to_num(hi.float(), nan=math.inf)[..., None, None]
        return torch.minimum(torch.maximum(q, lo), hi)
