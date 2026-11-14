# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

"""Forecast rollout, transformed views, post-processing and variable-length batching."""

import math
import operator

import numpy as np

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
F = _safe_import("torch.nn.functional")

from .covariates import prepare_covariates, usable_covariates, xreg_fit
from .model import MAX_VARIATES, QUANTILES, SKIP_TOKENS

LEVELS = torch.tensor(QUANTILES, dtype=torch.float64)
# Per-pass limits on trunk tokens and input steps, summed over all rows and variates.
FORECAST_TOKENS = 1 << 19
FORECAST_STEPS = 1 << 26


def rollout(
    model,
    history: torch.Tensor,
    horizon: int,
    series_ids: torch.Tensor,
    mixed: bool,
    covariates: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out median paths from history [B,V,T] to quantiles [B,V,horizon,9].
    T >= context_length; optional covariates [B,C,T+horizon] align with history and horizon."""
    c = model.config
    context = c.context_length
    if covariates is not None:
        end = history.shape[-1]
        future = math.ceil(c.prediction_length / c.patch_length) * c.patch_length
        covariates = F.pad(covariates, (0, future), value=float("nan"))
    width, fit = history.shape[1], False
    if covariates is not None:
        fit = c.fit_row and width == 1
        width += covariates.shape[1] + 1
    blocks, remaining = [], horizon
    while remaining:
        window = history[..., -context:]
        cov = (
            None
            if covariates is None
            else covariates[..., end - context : end + future]
        )
        steps = min(remaining, c.prediction_length)
        tokens, length = pass_size(c, window, cov, steps, width, fit)
        rows = max(1, min(FORECAST_TOKENS // tokens, FORECAST_STEPS // length))
        prediction = torch.cat(
            [
                model(
                    window[i : i + rows],
                    series_ids[i : i + rows],
                    mixed,
                    None if cov is None else cov[i : i + rows],
                    steps=steps,
                )
                for i in range(0, len(window), rows)
            ]
        )
        take = min(remaining, prediction.shape[-2])
        if covariates is not None:
            end += take
        blocks.append(prediction[..., :take, :])
        history = torch.cat([history, prediction[..., 4]], dim=-1)[..., -context:]
        remaining -= take
    return torch.cat(blocks, dim=-2)


def pass_size(
    c,
    window: torch.Tensor,
    covariates: torch.Tensor | None,
    steps: int,
    width: int,
    fit: bool,
):
    """Return (patch tokens, raw steps) per batch row for window [B,V,context_length].
    Count `width` rows after skipping missing prefixes; `fit` adds one full-window row and the
    other rows' inputs up to it."""
    context, patch = c.context_length, c.patch_length
    seen = torch.isfinite(window).flatten(0, -2).any(0)
    if (
        covariates is not None
    ):  # known-future rows hold each covariate one patch earlier
        cov = torch.isfinite(covariates[..., :context]).flatten(0, -2).any(0)
        seen |= cov | F.pad(cov[patch:], (0, patch), value=False)
    position = torch.arange(context, device=window.device)
    first = int(torch.where(seen, position, context - 1).amin())
    far_len = max(0, context - c.near_context)
    far = math.ceil(max(0, far_len - first) / (patch * c.far_pool))
    near = math.ceil((context - max(first, far_len)) / patch)
    unit = SKIP_TOKENS * patch * c.far_pool
    future = math.ceil(c.prediction_length / patch) * patch
    length = context - min(first, far_len) // unit * unit + future
    tokens = far + near + math.ceil(steps / patch)
    if not fit:
        return width * tokens, width * length
    whole = (
        far_len // (patch * c.far_pool)
        + (context - far_len) // patch
        + math.ceil(steps / patch)
    )
    # An accepted fit row spans the window. The trunk computes and stores the other rows from
    # their own start, but their input tokens and rotary tables are padded up to the fit row:
    # about 2 KB per empty token on a GPU, a tenth of a computed token of the smallest model.
    # They are charged a quarter of a token, which also keeps the step-domain work in bounds.
    if not torch.isfinite(window).flatten(1).any(1).all():
        return width * whole, width * (
            context + future
        )  # an unobserved item's rows run whole
    spent = (width - 1) * tokens + whole
    return spent + (width * whole - spent) // 4, (width - 1) * length + context + future


def flip_back(q: torch.Tensor) -> torch.Tensor:
    """Quantiles of -y from quantiles of y."""
    return -q.flip(-1)


def squared(q: torch.Tensor) -> torch.Tensor:
    """Quantiles of y from quantiles of sqrt(y) (the square-root views; sign kept)."""
    return (q.double().sign() * q.double() ** 2).float()


def median_of_views(members: list) -> torch.Tensor:
    """Per quantile, the mean of the middle two of the four views' sorted quantiles."""
    return torch.stack([m.sort(-1).values for m in members]).sort(0).values[1:3].mean(0)


def pool(history: torch.Tensor, k: int, points: int) -> torch.Tensor:
    """NaN-mean pool the last k * points steps of [B,V,T] into [B,V,points] block means, the last
    block ending at the forecast origin; empty blocks are NaN."""
    raw = history[..., -k * points :]
    short = k * points - raw.shape[-1]
    if short > 0:
        raw = F.pad(raw, (short, 0), value=float("nan"))
    blocks = raw.unflatten(-1, (points, k))
    observed = torch.isfinite(blocks)
    count = observed.sum(-1)
    total = torch.where(observed, blocks, 0).sum(-1)
    mean = total / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


def interpolate(coarse: torch.Tensor, k: int, horizon: int) -> torch.Tensor:
    """Coarse forecasts [B,V,Hc,Q] of block means -> raw leads [B,V,H,Q]: coarse point j sits at
    raw lead (j + 1/2) k - 1/2, linear in between, nearest value outside."""
    count = coarse.shape[-2]
    lead = torch.arange(horizon, device=coarse.device, dtype=torch.float64)
    u = ((lead - (k - 1) / 2) / k).clamp(0, count - 1)
    low = u.floor().long().clamp(max=count - 1)
    high = (low + 1).clamp(max=count - 1)
    frac = (u - low.double()).float()[:, None]
    return coarse[..., low, :] * (1 - frac) + coarse[..., high, :] * frac


def linear_pool(members: list) -> torch.Tensor:
    """Return [...,9] quantiles of an equal-weight mixture of piecewise-linear quantile CDFs.
    Tails reach probabilities 0 and 1 one outer-quantile spacing beyond each endpoint."""
    qs = [m.double().sort(-1).values for m in members]
    levels = torch.cat([LEVELS.new_zeros(1), LEVELS, LEVELS.new_ones(1)]).to(
        qs[0].device
    )
    knots = []
    for q in qs:
        lo = q[..., :1] - (q[..., 1:2] - q[..., :1])
        hi = q[..., 8:9] + (q[..., 8:9] - q[..., 7:8])
        k = torch.cat([lo, q, hi], -1)
        eps = 1e-9 * (
            k.abs().amax(-1, keepdim=True) + 1
        )  # Keep knots strictly increasing.
        knots.append(k + torch.arange(11, device=k.device) * eps)
    pts = torch.cat(knots, -1).sort(-1).values
    cdf = 0
    for k in knots:
        idx = torch.searchsorted(k.contiguous(), pts.contiguous(), right=True)
        i0 = (idx - 1).clamp(0, 10)
        i1 = idx.clamp(0, 10)
        k0, k1 = k.gather(-1, i0), k.gather(-1, i1)
        l0, l1 = levels[i0], levels[i1]
        frac = torch.where(k1 > k0, (pts - k0) / torch.where(k1 > k0, k1 - k0, 1), 0)
        val = l0 + frac * (l1 - l0)
        val = torch.where(idx == 0, 0.0, torch.where(idx >= 11, 1.0, val))
        cdf = cdf + val / len(knots)
    cdf = cdf.cummax(-1).values
    tau = LEVELS.to(pts.device).expand(*pts.shape[:-1], 9).contiguous()
    j = torch.searchsorted(cdf.contiguous(), tau, right=False).clamp(
        1, pts.shape[-1] - 1
    )
    f0, f1 = cdf.gather(-1, j - 1), cdf.gather(-1, j)
    p0, p1 = pts.gather(-1, j - 1), pts.gather(-1, j)
    fr = torch.where(f1 > f0, (tau - f0) / torch.where(f1 > f0, f1 - f0, 1), 0).clamp(
        0, 1
    )
    return (p0 + fr * (p1 - p0)).float()


def views_forecast(
    model,
    x: torch.Tensor,
    horizon: int,
    series_ids: torch.Tensor,
    mixed: bool,
    covariates: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine transformed views of history [B,V,context_length] into quantiles [B,V,H,9].
    Optional covariates [B,C,context_length+H] align with history and horizon."""
    c = model.config
    context, pred = c.context_length, c.prediction_length
    first = min(horizon, pred)
    observed = torch.isfinite(x)
    nonneg = observed.any(-1) & torch.where(observed, x >= 0, True).all(-1)  # [B,V]

    def run(values, steps, ids=series_ids, cov=None):
        if cov is not None:
            cov = cov[..., : values.shape[-1] + steps]
        return rollout(model, values, steps, ids, mixed, cov)

    rows = nonneg.any(-1)
    if horizon <= pred:
        # Adjacent views share period detection and covariate computation.
        root = torch.where(torch.isfinite(x), x.clamp_min(0).sqrt(), float("nan"))
        batch = x.shape[0]
        views = torch.stack([x, -x, root, -root], 1).flatten(0, 1)
        every = torch.ones_like(rows)
        take = torch.stack([every, every, rows, rows], 1).flatten()
        q = run(
            views[take],
            first,
            series_ids.repeat_interleave(4, 0)[take],
            None if covariates is None else covariates.repeat_interleave(4, 0)[take],
        )
        out = q.new_full((batch * 4, *q.shape[1:]), float("nan"))
        out[take] = q
        out = out.unflatten(0, (batch, 4))
        native, flip = out[:, 0], flip_back(out[:, 1])
        members = [native, flip]
        keep = nonneg[..., None, None]
        for q, fallback in ((out[:, 2], native), (flip_back(out[:, 3]), flip)):
            members.append(torch.where(keep, squared(q), fallback))
        return median_of_views(members).sort(-1).values
    native_full = run(x, horizon, cov=covariates)
    native = native_full[..., :first, :]
    flip = flip_back(run(-x, first, cov=covariates))
    members = [native, flip]
    if rows.any():
        xs = x[rows]
        root = torch.where(torch.isfinite(xs), xs.clamp_min(0).sqrt(), float("nan"))
        ids = series_ids[rows]
        keep = nonneg[rows][..., None, None]
        cr = None if covariates is None else covariates[rows]
        for q, fallback in (
            (run(root, first, ids, cr), native),
            (flip_back(run(-root, first, ids, cr)), flip),
        ):
            full = fallback.clone()
            full[rows] = torch.where(keep, squared(q), fallback[rows])
            members.append(full)
    else:
        members += [native, flip]
    near = median_of_views(members)
    if horizon <= pred:
        return near.sort(-1).values
    rest = horizon - pred
    paths = torch.cat(
        [torch.cat([x, native[..., p]], -1)[..., -context:] for p in (1, 7)]
    )
    tail = [
        native_full[..., pred:, :],
        *run(paths, rest, series_ids.repeat(2, 1)).chunk(2),
    ]
    pooled = pool(x, 2, context)
    steps = math.ceil(horizon / 2)
    coarse = 0.5 * (run(pooled, steps) + flip_back(run(-pooled, steps)))
    tail.append(interpolate(coarse, 2, horizon)[..., pred:, :])
    far = linear_pool(tail)
    return torch.cat([near, far], -2).sort(-1).values


def clip_to_sign(history: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Clip [B,V,H,Q] forecasts to the sign of a one-signed [B,V,T] history."""
    seen = torch.isfinite(history)
    low = torch.where(seen, history, math.inf).amin(-1)
    high = torch.where(seen, history, -math.inf).amax(-1)
    any_seen = seen.any(-1)
    nonneg = (any_seen & (low >= 0))[..., None, None]
    nonpos = (any_seen & (high <= 0) & (low < 0))[..., None, None]
    result = torch.where(nonneg, result.clamp_min(0), result)
    return torch.where(nonpos, result.clamp_max(0), result)


def integer_snap(history: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """Round forecasts for variates with non-negative integer-valued observed histories."""
    seen = torch.isfinite(history)
    integral = torch.where(seen, history == history.round(), True).all(-1)
    nonneg = torch.where(seen, history >= 0, True).all(-1)
    flag = integral & nonneg & seen.any(-1)
    return torch.where(flag[..., None, None], result.round(), result)


def _float32(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(torch.float32)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def stack_covariates(covariates, batch: int, length: int, device):
    """Stack covariates into float32 [B,C,length], padding with NaN; return None when empty.
    Accept a tensor/array [B,C,length] or B entries of None or [C_b,length]."""
    if covariates is None:
        return None
    if torch.is_tensor(covariates) or isinstance(covariates, np.ndarray):
        covariates = list(covariates)
    rows = [None if c is None else _float32(c) for c in covariates]
    if len(rows) != batch:
        raise ValueError("covariates must have one entry per batch row")
    width = max((len(r) for r in rows if r is not None), default=0)
    if not width:
        return None
    out = torch.full((batch, width, length), float("nan"), device=device)
    for b, r in enumerate(rows):
        if r is None:
            continue
        if r.ndim != 2 or r.shape[-1] != length:
            raise ValueError(
                f"covariates row {b}: expected [C, {length}], got {tuple(r.shape)}"
            )
        out[b, : len(r)] = r
    return out


def backtest_loss(truth: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Scaled pinball loss per batch row of q [B,V,H,9] against truth [B,V,H] (NaN skipped)."""
    levels = LEVELS.to(q.device)
    ok = torch.isfinite(truth)
    err = torch.where(ok, truth, 0)[..., None] - q
    pin = torch.maximum(levels * err, (levels - 1) * err).mean(-1) * ok
    return pin.sum((1, 2)) / torch.where(ok, truth.abs(), 0).sum((1, 2)).clamp_min(1e-9)


def _post_process(model, history, result):
    c = model.config
    if c.sign_clip:
        result = clip_to_sign(history, result)
    if c.integer_snap:
        result = integer_snap(history, result)
    return result


def covariate_variants(
    model,
    history: torch.Tensor,
    horizon: int,
    series_ids: torch.Tensor,
    covariates: torch.Tensor,
    rows: torch.Tensor,
    xreg: torch.Tensor,
    views: bool,
    mixed: bool,
) -> torch.Tensor:
    """Forecast selected history rows [J,V,T] as [J,V,H,9], with per-row XReg flags [J].
    Covariates [B,C,T+horizon] align with history [B,V,T]; T <= context_length."""
    c = model.config
    context = c.context_length
    left = max(0, context - history.shape[-1])
    history = F.pad(history, (left, 0), value=float("nan"))
    covariates = F.pad(covariates, (left, 0), value=float("nan"))
    original = values = history[rows]
    fitted = None
    if bool(xreg.any()):
        future = torch.zeros(
            covariates.shape[-1], dtype=torch.bool, device=history.device
        )
        future[-horizon:] = True
        full = F.pad(history, (0, horizon), value=float("nan"))
        fitted = xreg_fit(full, covariates, future, c.xreg_gate)[rows]
        values = torch.where(
            xreg[:, None, None], original - fitted[..., : original.shape[-1]], original
        )
    cov, ids = covariates[rows], series_ids[rows]
    if views:
        result = views_forecast(model, values, horizon, ids, mixed, cov)
    else:
        result = rollout(model, values, horizon, ids, mixed, cov)
    if fitted is not None:
        result = torch.where(
            xreg[:, None, None, None],
            result + fitted[..., -horizon:, None].to(result.dtype),
            result,
        )
    return _post_process(model, original, result)


def _backtest_covariates(covariates: torch.Tensor, horizon: int) -> torch.Tensor:
    """Hide past-only covariates over the backtest horizon as well as the real horizon."""
    past = covariates[..., :-horizon].clone()
    later = ~torch.isfinite(covariates[..., -horizon:]).any(-1)
    past[..., -horizon:] = torch.where(
        later[..., None], float("nan"), past[..., -horizon:]
    )
    return past


def _covariate_rules(model, history, horizon, series_ids, covariates, views, mixed):
    """Select XReg per row using history length and a held-out pinball-loss comparison."""
    c = model.config
    batch, device = history.shape[0], history.device
    observed = torch.isfinite(history).sum(-1).amin(-1)
    if not c.xreg:
        long = torch.ones(batch, dtype=torch.bool, device=device)
    elif c.rows_only_history > 0:
        long = observed > c.rows_only_history
    else:
        long = torch.zeros(batch, dtype=torch.bool, device=device)
    guard = ~long & (observed >= max(c.xreg_guard_min, horizon + 64))
    if not c.xreg_guard:
        guard = torch.zeros_like(guard)
    index = torch.arange(batch, device=device)
    # Slots 0/1 use full history with/without XReg; 2/3 use the backtest at T - horizon.
    parts = [
        (index[~long], 0),
        (index[guard | long], 1),
        (index[guard], 2),
        (index[guard], 3),
    ]
    rows = torch.cat([r + (batch if slot >= 2 else 0) for r, slot in parts])
    slots = torch.cat([torch.full_like(r, slot) for r, slot in parts])
    item = rows % batch
    order = torch.argsort(item * 4 + slots)
    rows, slots, item = rows[order], slots[order], item[order]
    if bool(guard.any()):
        cut = F.pad(history[..., :-horizon], (horizon, 0), value=float("nan"))
        past = F.pad(
            _backtest_covariates(covariates, horizon), (horizon, 0), value=float("nan")
        )
        history2 = torch.cat([history, cut])
        covariates2 = torch.cat([covariates, past])
        ids2 = torch.cat([series_ids, series_ids])
    else:
        history2, covariates2, ids2 = history, covariates, series_ids
    out = covariate_variants(
        model, history2, horizon, ids2, covariates2, rows, slots % 2 == 0, views, mixed
    )

    def part(slot):
        mask = slots == slot
        return item[mask], out[mask]

    result = out.new_empty((batch, *out.shape[1:]))
    rows_x, with_x = part(0)
    result[rows_x] = with_x
    rows_n, no_x = part(1)
    result[rows_n[long[rows_n]]] = no_x[long[rows_n]]
    if bool(guard.any()):
        rows_g, bt_x = part(2)
        _, bt_n = part(3)
        truth = history[rows_g][..., -horizon:].cpu().double()
        use = backtest_loss(
            truth, bt_n.cpu().double()
        ) < c.xreg_guard_margin * backtest_loss(truth, bt_x.cpu().double())
        result[rows_g] = torch.where(
            use.to(device)[:, None, None, None], no_x[guard[rows_n]], result[rows_g]
        )
    return result


@torch.inference_mode()
def forecast(
    model,
    history,
    horizon: int,
    series_ids=None,
    covariates=None,
    views: bool | None = None,
    mixed: bool | None = None,
) -> torch.Tensor:
    """[B,T]/[B,V,T] history -> [B,H,9]/[B,V,H,9] quantiles; NaN = missing, V <= 32.
    Equal series_ids [B,V] group variates; covariates [B,C,T+horizon] or B None/[C_b,T+horizon].
    Read <= context_length steps and 16 usable covariates; rescale if std is below about 1e-4."""
    if not torch.is_tensor(history):
        history = _float32(history)
    horizon = operator.index(horizon)
    if history.ndim not in (2, 3) or 0 in history.shape or horizon <= 0:
        raise ValueError(
            "Provide nonempty [B,T] or [B,V,T] history and a positive horizon"
        )
    univariate = history.ndim == 2
    if univariate:
        history = history[:, None]
    if history.shape[1] > MAX_VARIATES:
        raise ValueError(
            f"Split related variates into groups of at most {MAX_VARIATES}"
        )
    device = next(model.parameters()).device
    history = history.to(device=device, dtype=torch.float32)
    if series_ids is None:
        series_ids = torch.zeros(history.shape[:2], dtype=torch.long, device=device)
    series_ids = torch.as_tensor(series_ids, device=device)
    if series_ids.shape != history.shape[:2]:
        raise ValueError("series_ids must match the batch and variate dimensions")
    c = model.config
    context = c.context_length
    if covariates is not None and not c.covariates:
        raise ValueError(
            "this checkpoint has no covariate weights; forecast without covariates"
        )
    covariates = stack_covariates(
        covariates, history.shape[0], history.shape[-1] + horizon, device
    )
    history = history[..., -context:]
    if covariates is not None:
        covariates = usable_covariates(
            history, covariates[..., -(history.shape[-1] + horizon) :], horizon
        )
    if covariates is not None:
        several = (series_ids != series_ids[:, :1]).any(-1)
        if bool((several & torch.isfinite(covariates).flatten(1).any(-1)).any()):
            raise ValueError(
                "a row with covariates must hold one series (one series id); put "
                "each series in its own row"
            )
    views = c.views if views is None else views
    mixed = history.is_cuda if mixed is None else mixed and history.is_cuda
    was_training = model.training
    model.eval()
    try:
        if covariates is not None:
            result = _covariate_rules(
                model, history, horizon, series_ids, covariates, views, mixed
            )
        else:
            history = F.pad(
                history, (context - history.shape[-1], 0), value=float("nan")
            )
            if views:
                result = views_forecast(model, history, horizon, series_ids, mixed)
            else:
                result = rollout(model, history, horizon, series_ids, mixed)
            result = _post_process(model, history, result)
    finally:
        model.train(was_training)
    observed = torch.isfinite(history).any(
        -1
    )  # [B,V]: a variate with no observed value gets 0
    result = torch.where(observed[..., None, None], result, 0.0)
    return result[:, 0] if univariate else result


def pack(histories: list, context: int):
    """Pack [V,T] histories into (history [R,W,context], ids [R,W], input (row,slot) positions).
    V and W <= 32; histories are right-aligned and cropped; id -1 marks padding."""
    rows, row, used = [], [], 0
    for index, h in enumerate(histories):
        if used + len(h) > MAX_VARIATES:
            rows.append(row)
            row, used = [], 0
        row.append(index)
        used += len(h)
    rows.append(row)
    width = max(sum(len(histories[i]) for i in row) for row in rows)
    batch = torch.full((len(rows), width, context), float("nan"))
    ids = torch.full((len(rows), width), -1, dtype=torch.long)
    places = [None] * len(histories)
    for r, row in enumerate(rows):
        offset = 0
        for index in row:
            h = histories[index]
            n = min(context, h.shape[-1])
            batch[r, offset : offset + len(h), context - n :] = torch.from_numpy(
                h[:, -n:]
            )
            ids[r, offset : offset + len(h)] = index
            places[index] = (r, offset)
            offset += len(h)
    return batch, ids, places


def forecast_series(
    model,
    series: list,
    horizon: int,
    batch_size: int = 1024,
    future_covariates: list | None = None,
    past_covariates: list | None = None,
    shared_categories: bool = True,
    **options,
) -> list:
    """[T]/[V,T] histories -> float32 CPU numpy [H,9]/[V,H,9] forecasts; NaN = missing.
    Future/past covariates [C,T+horizon]/[C,T] align by position; >32 variates split into groups.
    `shared_categories=False` codes history and horizon separately; options go to `forecast`."""
    for s in series:
        if np.ndim(s) not in (1, 2) or 0 in np.shape(s):
            raise ValueError("Each series must be a nonempty [T] or [V,T] array")
    if future_covariates is None and past_covariates is None:
        return _forecast_packed(model, series, horizon, batch_size, **options)
    for name, given in (
        ("future_covariates", future_covariates),
        ("past_covariates", past_covariates),
    ):
        if hasattr(given, "items"):
            raise ValueError(
                f"{name} takes a list with one entry per series, got a "
                f"{type(given).__name__}; for one series, wrap its covariates in a "
                f"list: {name}=[...]"
            )
    count = len(series)
    future_covariates = (
        [None] * count if future_covariates is None else list(future_covariates)
    )
    past_covariates = (
        [None] * count if past_covariates is None else list(past_covariates)
    )
    if len(future_covariates) != count or len(past_covariates) != count:
        raise ValueError(
            "future_covariates / past_covariates need one entry per series"
        )
    items = []
    for s, future, past in zip(series, future_covariates, past_covariates):
        s = np.asarray(s, dtype=np.float32)
        targets = s[None] if s.ndim == 1 else s
        items.append(
            (
                targets,
                prepare_covariates(
                    targets.shape[-1],
                    horizon,
                    future,
                    past,
                    shared_categories=shared_categories,
                ),
            )
        )
    outputs = forecast_items(model, items, horizon, batch_size, **options)
    return [out[0] if np.ndim(s) == 1 else out for s, out in zip(series, outputs)]


def forecast_items(
    model, items: list, horizon: int, batch_size: int = 256, **options
) -> list:
    """Return [V,H,9] forecasts per item with targets [V,T] and covariates [C,T+horizon] or None.
    Related variates share covariates and are split into groups of at most 32."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    context = model.config.context_length
    rows = []
    for index, (targets, covs) in enumerate(items):
        bad = np.ndim(targets) != 2 or 0 in np.shape(targets)
        if not bad and covs is not None:
            bad = (
                np.ndim(covs) != 2
                or np.shape(covs)[-1] != np.shape(targets)[-1] + horizon
            )
        if bad:
            raise ValueError(
                f"item {index}: expected nonempty targets [V, T] and covariates "
                f"[C, T + {horizon}] or None"
            )
        for first in range(0, len(targets), MAX_VARIATES):
            rows.append((index, first, targets[first : first + MAX_VARIATES], covs))
    out = [
        np.zeros((len(t), horizon, len(QUANTILES)), dtype=np.float32) for t, _ in items
    ]
    groups = {}
    for row in rows:
        groups.setdefault(row[2].shape[0], []).append(row)
    for variates, group in groups.items():
        for begin in range(0, len(group), batch_size):
            chunk = group[begin : begin + batch_size]
            width = min(context, max(r[2].shape[-1] for r in chunk))
            history = np.full((len(chunk), variates, width), np.nan, dtype="float32")
            covs = []
            for b, (_, _, block, c) in enumerate(chunk):
                w = min(width, block.shape[-1])
                history[b, :, width - w :] = block[:, block.shape[-1] - w :]
                if c is None:
                    covs.append(None)
                    continue
                full = np.full((len(c), width + horizon), np.nan, dtype="float32")
                full[:, width - w :] = c[:, block.shape[-1] - w :]
                covs.append(full)
            cov_arg = covs if any(c is not None for c in covs) else None
            prediction = (
                forecast(
                    model,
                    torch.from_numpy(history),
                    horizon,
                    covariates=cov_arg,
                    **options,
                )
                .cpu()
                .numpy()
            )
            for b, (index, first, block, _) in enumerate(chunk):
                out[index][first : first + len(block)] = prediction[b]
    return out


def _forecast_packed(
    model, series: list, horizon: int, batch_size: int, **options
) -> list:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    context = model.config.context_length
    results = []
    for start in range(0, len(series), batch_size):
        chunk = [
            np.ascontiguousarray(s, dtype=np.float32)
            for s in series[start : start + batch_size]
        ]
        blocks, owner = [], []
        for index, s in enumerate(chunk):
            s = s[None] if s.ndim == 1 else s
            for first in range(0, len(s), MAX_VARIATES):
                blocks.append(s[first : first + MAX_VARIATES])
                owner.append(index)
        outputs = []
        for first in range(0, len(blocks), batch_size):
            group = blocks[first : first + batch_size]
            history, ids, places = pack(group, context)
            prediction = forecast(model, history, horizon, ids, **options).cpu().numpy()
            outputs += [
                prediction[r, o : o + len(h)] for (r, o), h in zip(places, group)
            ]
        for index, s in enumerate(chunk):
            parts = [out for out, i in zip(outputs, owner) if i == index]
            stacked = np.concatenate(parts)
            results.append(stacked[0] if s.ndim == 1 else stacked)
    return results
