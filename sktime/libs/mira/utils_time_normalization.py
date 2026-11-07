# ruff: noqa
# Copyright (c) Microsoft
# Licensed under MIT

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


def safe_nanmin(x: torch.Tensor, dim=1, keepdim=True):
    mask = torch.isnan(x)
    if mask.any():
        x = x.clone()
        x[mask] = torch.inf
    values, _ = torch.min(x, dim=dim, keepdim=keepdim)
    return values


def safe_nanmax(x: torch.Tensor, dim=1, keepdim=True):
    mask = torch.isnan(x)
    if mask.any():
        x = x.clone()
        x[mask] = -torch.inf
    values, _ = torch.max(x, dim=dim, keepdim=keepdim)
    return values


def enforce_unique_monotonic(times: torch.Tensor, eps: float = 1e-4):
    """
    Ensure strictly increasing timestamps without destroying structure.
    Only fix ties or backwards by adding eps.
    """
    out = times.clone()
    B, L = out.shape

    for b in range(B):
        for i in range(1, L):
            if out[b, i] <= out[b, i - 1]:
                out[b, i] = out[b, i - 1] + eps
    return out


def snap_values(t: torch.Tensor, snap: str = "decimal", snap_step: float = 0.1):
    """
    Snap values to a grid:
      snap = "integer" → round to nearest integer
      snap = "decimal" → round to nearest k * snap_step
      snap = None      → no snapping
    """
    if snap is None:
        return t

    if snap == "integer":
        return torch.round(t)

    if snap == "decimal":
        return torch.round(t / snap_step) * snap_step

    raise ValueError(f"Unknown snapping mode: {snap}")


def normalize_time_for_ctrope(
    time_values: torch.Tensor,
    attention_mask: torch.Tensor = None,
    seq_length: int = None,
    alpha: float = 1.0,
    snap: str = "decimal",  # "integer", "decimal", or None
    snap_step: float = 0.1,  # Only used when snap="decimal"
):
    """
    Normalize raw time_values → CT-RoPE scale, then apply snapping, then enforce monotonicity.
    """

    time_values = time_values.to(torch.float32)

    if attention_mask is not None:
        masked_time = time_values.masked_fill(attention_mask == 0, float("nan"))
    else:
        masked_time = time_values

    t_min = safe_nanmin(masked_time, dim=1, keepdim=True)
    t_max = safe_nanmax(masked_time, dim=1, keepdim=True)
    denom = (t_max - t_min).clamp(min=1e-8)

    t_norm = (time_values - t_min) / denom

    max_range = alpha * float(seq_length - 1)
    t_scaled = t_norm * max_range

    t_scaled = torch.nan_to_num(t_scaled, nan=0.0)

    # Snapping
    t_snapped = snap_values(t_scaled, snap=snap, snap_step=snap_step)

    t_fixed = enforce_unique_monotonic(t_snapped)

    return t_fixed, t_min, t_max
