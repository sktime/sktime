"""Normalization for Timestral: causal per-patch mean/std scaling with an asinh squash.

Each patch is normalized with the mean and std of every observed value up to the end of
that patch, so the context never sees future statistics. Forecasts come back in the
normalized (asinh) space of the last context patch and are inverted with `sinh`.
"""

import functools

import torch
from torch import Tensor


def compute_causal_statistics(
    data: Tensor,
    observation_mask: Tensor,
    minimum_scale: float,
    stabilize_with_global: bool,
    scale_factor_exponent: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Causal running mean and std along the last axis (Welford form, in float64).

    Returns `(means, scales)`, both shaped like `data`; position `t` uses the observed
    values at positions `0..t` only. Unobserved positions get zero weight. The variance
    uses Bessel's correction and the std is clamped to `minimum_scale` after the sqrt.
    """
    weights = torch.ones_like(data) * observation_mask
    with torch.no_grad():
        x = data.to(torch.float64)
        w = weights.to(torch.float64)
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        if prev_deterministic and data.device.type == "cuda":
            torch.use_deterministic_algorithms(False)
        try:
            weighted = w * x
            cum_w = torch.cumsum(w, dim=-1)
            cum_x = torch.cumsum(weighted, dim=-1)
            denominator = cum_w.clamp_min(1.0)
            means = cum_x / denominator
            shifted = torch.zeros_like(means)
            shifted[..., 1:] = means[..., :-1]
            increment = (x - shifted) * (x - means) * w
            m2 = torch.cumsum(increment, dim=-1)
            variance = m2 / torch.clamp(denominator - 1.0, min=1.0)
            scales = torch.sqrt(variance).clamp(min=minimum_scale)
            if stabilize_with_global:
                lo, hi = 10.0 ** (-scale_factor_exponent), 10.0**scale_factor_exponent
                global_den = (weights * observation_mask).sum(-1, keepdim=True).clamp_min(1.0)
                global_means = torch.nan_to_num(weighted.sum(-1, keepdim=True) / global_den)
                global_var = (((x - global_means) * weights * observation_mask) ** 2).sum(-1, keepdim=True) / global_den
                global_scale = torch.sqrt(global_var + minimum_scale).expand_as(scales)
                floor = torch.max(torch.tensor(minimum_scale, device=scales.device, dtype=scales.dtype), global_scale * lo)
                scales = torch.clamp(scales, min=floor, max=global_scale * hi)
            means = means.to(data.dtype)
            scales = scales.to(data.dtype)
        finally:
            if prev_deterministic and data.device.type == "cuda":
                torch.use_deterministic_algorithms(True)
    return means, scales


def causal_patch_loc_scale(
    data: Tensor,
    observation_mask: Tensor,
    patch_size: int,
    minimum_scale: float,
    stabilize_with_global: bool,
) -> tuple[Tensor, Tensor]:
    """Per-patch `(loc, scale)` broadcast back to every time step of the patch.

    Each patch uses the causal statistics at its last time step, i.e. everything
    observed up to the end of that patch.
    """
    assert data.shape[-1] % patch_size == 0, f"length {data.shape[-1]} is not a multiple of patch_size {patch_size}"
    with torch.no_grad():
        means, scales = compute_causal_statistics(data, observation_mask, minimum_scale, stabilize_with_global)
        patch_means = means.unfold(-1, patch_size, patch_size)[..., -1]
        patch_scales = scales.unfold(-1, patch_size, patch_size)[..., -1]
        return patch_means.repeat_interleave(patch_size, dim=-1), patch_scales.repeat_interleave(patch_size, dim=-1)


def _fp32_io(*dtypes: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    """`(out_dtype, work_dtype)`: compute in at least fp32, return the promoted input dtype."""
    out_dtype = functools.reduce(torch.promote_types, dtypes)
    return out_dtype, torch.promote_types(out_dtype, torch.float32)


def apply_affine_asinh_masked(
    data: Tensor,
    loc: Tensor,
    scale: Tensor,
    observation_mask: Tensor,
    use_asinh: bool,
) -> Tensor:
    """`asinh((data - loc) / scale)` on observed positions; unobserved positions become 0."""
    out_dtype, work = _fp32_io(data.dtype, loc.dtype, scale.dtype)
    scaled = (data.to(work) - loc.to(work)) / scale.to(work).clamp_min(torch.finfo(work).eps)
    scaled = scaled.masked_fill(~observation_mask, 0.0)
    if use_asinh:
        scaled = torch.where(observation_mask, scaled.asinh(), scaled)
    return scaled.to(out_dtype)


def invert_affine_asinh(x: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool) -> Tensor:
    """Inverse of the forward transform: `sinh(x) * scale + loc`."""
    out_dtype, work = _fp32_io(x.dtype, loc.dtype, scale.dtype)
    x = x.to(work)
    if use_asinh:
        x = x.sinh()
    return (x * scale.to(work) + loc.to(work)).to(out_dtype)


def clamp_nonfinite(vals: Tensor) -> Tensor:
    """Map +/-inf (a sinh overflow) to the nearest finite value along the last axis.

    A position whose whole last axis is non-finite becomes 0. NaN is left untouched so
    that real instability still surfaces downstream.
    """
    clamped = torch.where(
        vals == float("inf"),
        torch.where(vals.isfinite(), vals, -float("inf")).amax(dim=-1, keepdim=True),
        torch.where(
            vals == -float("inf"),
            torch.where(vals.isfinite(), vals, float("inf")).amin(dim=-1, keepdim=True),
            vals,
        ),
    )
    return torch.where(clamped.isinf(), torch.zeros_like(clamped), clamped)


def denormalize_quantiles(preds: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool) -> Tensor:
    """Invert the normalization and clamp infinities; monotone quantiles stay in order."""
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def welford_combine(
    w_a: Tensor, mean_a: Tensor, m2_a: Tensor, w_b: Tensor, mean_b: Tensor, m2_b: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Merge two Welford states `(weight, mean, m2)` element-wise; `(0, 0, 0)` is the identity."""
    w = w_a + w_b
    w_safe = w.clamp(min=torch.finfo(w.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + delta.square() * w_a * w_b / w_safe
    is_zero = w == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    return w, mean, m2


def cumulative_welford(w: Tensor, mean: Tensor, m2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Inclusive prefix scan of Welford states along dim 1 (Hillis-Steele, `log2(N)` merges)."""
    n = w.shape[1]
    d = 1
    while d < n:
        pad = (d, 0)
        w_l = torch.nn.functional.pad(w[:, : n - d], pad)
        mean_l = torch.nn.functional.pad(mean[:, : n - d], pad)
        m2_l = torch.nn.functional.pad(m2[:, : n - d], pad)
        w, mean, m2 = welford_combine(w_l, mean_l, m2_l, w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(
    context: Tensor, observation_mask: Tensor, sinh_median: Tensor, minimum_scale: float
) -> tuple[Tensor, Tensor]:
    """Per-patch `(loc, scale)` of a decoded block in the advancing refine frame, each `(R, H)` float64.

    Horizon patch `j` is normalized with the statistics of the observed context plus the
    block's medians of patches `0..j-1`, each median denormalized in its own frame before
    it is folded in. With perfect medians this is causal per-patch normalization of the true
    series. The statistics are float64 Welford states, accumulated exactly as in training.

    Args:
        context: `(R, T)` block context, `T` a multiple of the patch size.
        observation_mask: `(R, T)`.
        sinh_median: `(R, H, patch_size)` the block's median in the last context patch's
            frame with the asinh undone, `(median - loc) / scale`.
        minimum_scale: Scale floor.
    """
    r, n_pred, ps = sinh_median.shape
    blank = sinh_median.new_zeros(r, n_pred, ps)
    # Per-patch moments of the context, scanned together with the zero-weight horizon patches.
    values = torch.cat([context.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    weights = torch.cat([observation_mask.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / cnt.clamp(min=1.0)
    m2 = (weights * (values - mean.unsqueeze(-1)).square()).sum(-1)
    is_zero = cnt == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))

    offset = mean  # medians are clamped relative to the context mean, so a large level survives
    clip = torch.finfo(torch.float32).max ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = (m2 / (cnt - 1.0).clamp(min=1.0)).clamp(min=0.0)
        scale = var.sqrt().clamp(min=minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None]).clamp(-clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = (folded - b_mean[:, None]).square().sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, torch.full_like(cnt, ps), b_mean, b_m2)
    return torch.stack(locs, dim=1), torch.stack(scales, dim=1)


def instance_norm_stats(values: Tensor, observation_mask: Tensor, minimum_scale: float) -> tuple[Tensor, Tensor, Tensor]:
    """Non-causal per-row mean/std over observed positions, for known-future covariates.

    Rows whose observed values are all 0 or 1 (indicator covariates) keep `loc=0`,
    `scale=1`. Returns `(loc, scale, is_binary)`, each `(R, 1)`.
    """
    orig_dtype = values.dtype
    stat_dtype = torch.promote_types(orig_dtype, torch.float32)
    x = values.to(stat_dtype)
    mask_f = observation_mask.to(stat_dtype)
    n = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    loc = (x * mask_f).sum(dim=-1, keepdim=True) / n
    var = (((x - loc) * mask_f) ** 2).sum(dim=-1, keepdim=True) / n
    scale = var.sqrt().clamp_min(minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(dim=-1, keepdim=True)
    is_binary = is_zero_one & observation_mask.any(dim=-1, keepdim=True)
    loc = torch.where(is_binary, torch.zeros_like(loc), loc)
    scale = torch.where(is_binary, torch.ones_like(scale), scale)
    return loc.to(orig_dtype), scale.to(orig_dtype), is_binary
