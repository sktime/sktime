# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

"""Detect dominant history periods and encode their phases in the slowest rotary pairs."""

import math

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")

MIN_OBSERVED = 64
LOCAL = 16
MAX_LOCAL = 128
EULER = 0.5772156649
TABLE_BLOCK = 256

_TABLES = {}


def _refine_table(length: int, device):
    key = (length, str(device))
    if key not in _TABLES:
        with torch.inference_mode(False), torch.no_grad():
            candidates = torch.arange(
                2, length // 2 + 1, device=device, dtype=torch.float32
            )
            steps = torch.arange(length, device=device, dtype=torch.float64)
            cos = torch.empty(
                length, candidates.numel(), device=device, dtype=torch.float32
            )
            sin = torch.empty_like(cos)
            # Blockwise fp64 angles avoid allocating a full fp64 DFT table.
            for start in range(0, candidates.numel(), TABLE_BLOCK):
                block = candidates[start : start + TABLE_BLOCK].double()
                angle = 2 * math.pi * torch.remainder(steps[:, None] / block[None], 1.0)
                cos[:, start : start + TABLE_BLOCK] = angle.cos()
                sin[:, start : start + TABLE_BLOCK] = angle.sin()
            _TABLES[key] = (candidates, cos, sin)
    return _TABLES[key]


def detect_periods(
    values: torch.Tensor,
    context: int,
    slots: int,
    alpha: float,
    local: float,
    patch: int,
) -> torch.Tensor:
    """Return [...,slots] dominant periods in steps (0: none) from values[..., :context].
    Input must contain at least `context` steps; periods dividing `patch` are excluded."""
    shape = values.shape[:-1]
    with torch.autocast(device_type=values.device.type, enabled=False):
        x = values[..., :context].reshape(-1, context).float()
        observed = torch.isfinite(x)
        weight = observed.float()
        count = weight.sum(-1, keepdim=True)
        safe = count.clamp_min(1)
        x0 = torch.where(observed, x, 0).double()
        w = weight.double()
        mean = x0.sum(-1, keepdim=True) / safe.double()
        centred = (x0 - mean) * w
        t = torch.arange(context, device=x.device, dtype=torch.float64)
        t_mean = (t * w).sum(-1, keepdim=True) / safe.double()
        tc = (t - t_mean) * w
        slope = (tc * centred).sum(-1, keepdim=True) / (tc * tc).sum(
            -1, keepdim=True
        ).clamp_min(1e-12)
        centred = (centred - slope * tc) * w
        std = (
            ((centred**2).sum(-1, keepdim=True) / safe.double()).sqrt().clamp_min(1e-8)
        )
        z = (centred / std).float()
        spectrum = torch.fft.rfft(z, n=context, dim=-1)
        power = spectrum.real**2 + spectrum.imag**2
        half = context // 2
        power = power[:, 1 : half + 1]
        total = power.sum(-1, keepdim=True).clamp_min(1e-12)
        intensity = power / total
        threshold = math.log(half / alpha) / half
        bins = torch.arange(1, half + 1, device=x.device, dtype=torch.float32)
        period = context / bins
        left = torch.nn.functional.pad(intensity[:, :-1], (1, 0), value=float("inf"))
        right = torch.nn.functional.pad(intensity[:, 1:], (0, 1), value=float("inf"))
        peak = (intensity > left) & (intensity > right) & (intensity > threshold)
        # Require at least 2.5 cycles in the observed history.
        longest = torch.floor(count / 2.5)
        peak &= (period >= 2) & (period <= longest)
        peak &= count >= MIN_OBSERVED
        # Exponential noise has E[log I] = log(mean) - Euler gamma.
        # Scale the window with zero-padding to cover short series' broad spectral peaks.
        logi = intensity.clamp_min(1e-12).log()
        w = (LOCAL * context / safe).round().clamp(LOCAL, MAX_LOCAL).long()
        csum = torch.nn.functional.pad(logi.cumsum(-1), (1, 0))
        k = torch.arange(half, device=x.device)[None]
        lo_i = (k - w).clamp_min(0)
        hi_i = (k + w + 1).clamp_max(half)
        background = (csum.gather(-1, hi_i) - csum.gather(-1, lo_i)) / (hi_i - lo_i)
        background = (background + EULER).exp()
        peak &= intensity > local * background
        score = torch.where(peak, intensity, torch.zeros_like(intensity))
        top, index = score.topk(min(3 * slots, half), dim=-1)
        candidates, cos, sin = _refine_table(context, x.device)
        if z.is_cuda:
            # TF32 only changes the argmax among near-tied neighboring integer periods.
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                exact = (z @ cos) ** 2 + (z @ sin) ** 2
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        else:
            exact = (z @ cos) ** 2 + (z @ sin) ** 2
        centre = (index.float() + 1)[..., None]
        offset = (context / candidates)[None, None] - centre
        near = offset.abs() < 1
        refined = torch.where(near, exact[:, None], torch.full_like(offset, -1)).argmax(
            -1
        )
        periods = candidates[refined]
        keep = top > 0
        # A period dividing the patch length has the same phase in every patch.
        keep &= torch.remainder(float(patch), periods) != 0
        same = periods[:, :, None] == periods[:, None, :]
        earlier = torch.ones_like(same[0]).tril(-1)
        keep &= ~(same & earlier & keep[:, None, :]).any(-1)
        order = torch.arange(keep.shape[-1], device=x.device).expand_as(keep)
        order = torch.where(keep, order, order + keep.shape[-1]).argsort(-1)[:, :slots]
        periods = torch.where(keep, periods, torch.zeros_like(periods)).gather(
            -1, order
        )
    return periods.reshape(*shape, slots)


def select_periods(
    config, values: torch.Tensor, history: int, far_len: int
) -> torch.Tensor:
    """Return [B,V,slots] periods for values [B,V,T] with `history` history steps.
    Use the detector span matching observed history; `far_len=0` selects full history."""

    def detect(x, context):
        return detect_periods(
            x,
            context,
            config.period_slots,
            config.period_alpha,
            config.period_local,
            config.patch_length,
        )

    if not far_len:
        return detect(values, history)
    near_history = history - far_len
    out = detect(values[..., far_len:], near_history)
    has_far = torch.isfinite(values[..., :far_len]).any(-1, keepdim=True)
    mid = min(config.detector_mid, history)
    mid_start = history - mid
    if mid > near_history and bool(has_far.any()):
        out = torch.where(has_far, detect(values[..., mid_start:], mid), out)
    if mid_start > 0:
        older = torch.isfinite(values[..., :mid_start]).any(-1, keepdim=True)
        if bool(older.any()):
            out = torch.where(older, detect(values, history), out)
    return out


def period_rotary(periods: torch.Tensor, starts: torch.Tensor, patch: int, half: int):
    """Return rotary cos/sin [N,1,S,half] for token starts [S] in raw steps (float64).
    The slowest `slots` pairs encode detected periods; missing periods have zero phase."""
    n, slots = periods.shape
    size = starts.shape[0]
    frequency = 10000 ** (
        -torch.arange(half, device=periods.device, dtype=torch.float32) / half
    )
    base = (starts / patch).float()[:, None] * frequency
    steps = starts.double()
    angle = base.expand(n, size, half).clone()
    p = periods.double()
    valid = p > 0
    safe = torch.where(valid, p, torch.ones_like(p))
    for k in range(slots):
        cycles = steps[None, :] / safe[:, k : k + 1]
        phase = 2 * math.pi * (cycles - torch.floor(cycles))
        phase = torch.where(valid[:, k : k + 1], phase, torch.zeros_like(phase))
        angle[:, :, half - slots + k] = phase.float()
    return angle.cos()[:, None], angle.sin()[:, None]
