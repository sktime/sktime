# Copyright 2026 Abel AI Lab.
# Source: https://github.com/Abel-ai-lab/Simons
# Apache-2.0; vendored from Abel-ai-lab/Simons at commit 8e338e915d26c8960713889ef12ab1cf5c08d591.
# Model weights are not bundled; see the Simons model license.
# ruff: noqa

"""Prepare known-future covariate rows and in-context ridge fits for forecasting."""

import numpy as np

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")

ROLE_TARGET, ROLE_KNOWN, ROLE_PAST, ROLE_FIT, ROLE_BINARY = 0, 1, 2, 3, 4
MAX_COVARIATES = 16


class CovariateModules(nn.Module):
    """Role vectors (zero for target rows) and the optional read block."""

    def __init__(self, config, block_type):
        super().__init__()
        self.role = nn.Parameter(
            torch.zeros(5 if config.binary_rows else 4, config.width)
        )
        self.read = block_type(config, causal=False) if config.read_after >= 0 else None

    def role_vectors(self, roles: torch.Tensor) -> torch.Tensor:
        """[B,V,W] additive role vectors (target rows get exactly 0)."""
        return self.role[roles] * (roles > 0)[..., None].to(self.role.dtype)

    def read_rows(self, x, related, batch: int, variates: int, size: int, roles):
        """x [B*V,S,W] -> x with target rows updated by the read (every other row unchanged)."""
        xm = (
            x.reshape(batch, variates, size, -1)
            .transpose(1, 2)
            .reshape(batch * size, variates, -1)
        )
        mask, update = read_mask(roles, related, size)
        y = read_forward(self.read, xm, mask)
        u = update.to(xm.dtype)
        out = xm + u * (y - xm)
        return (
            out.reshape(batch, size, variates, -1)
            .transpose(1, 2)
            .reshape(batch * variates, size, -1)
        )


def read_mask(roles, related, size: int):
    """(mask [B*S,1,V,V], update [B*S,V,1]) from roles [B,V] and related [B*S,1,V,V]: a target
    row reads itself and its group's known-future, indicator and fit rows."""
    variates = roles.shape[1]
    r = roles.repeat_interleave(size, 0)
    key = (r == ROLE_KNOWN) | (r == ROLE_FIT) | (r == ROLE_BINARY)
    cross = (r == ROLE_TARGET)[:, :, None] & key[:, None, :] & related[:, 0]
    eye = torch.eye(variates, dtype=torch.bool, device=roles.device)[None]
    return (cross | eye)[:, None], cross.any(-1)[..., None]


def read_forward(block, x, mask):
    """The read block (pre-norm attention + MLP) with an explicit fp32 softmax over few keys."""
    n, s, width = x.shape
    q, k, v = (
        block.qkv(block.attention_norm(x)).view(n, s, 3, block.heads, -1).unbind(2)
    )
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    weights = scores.float().masked_fill(~mask, float("-inf")).softmax(-1)
    attended = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(n, s, width)
    x = x + block.out(attended)
    return x + block.mlp(block.mlp_norm(x))


def ridge_fit(y: torch.Tensor, cov: torch.Tensor, use: torch.Tensor, skip: int = 0):
    """Fit y [B,L] on cov [B,C,L] selected by use [B,C], with a held-out R^2 check.
    Return (fit [B,skip+L] in target units, valid [B]); skipped missing steps get the intercept."""
    obs = torch.isfinite(y)
    w = obs.double()
    n = w.sum(-1, keepdim=True).clamp_min(1)
    yy = torch.where(obs, y, 0).double()
    mu = (yy * w).sum(-1, keepdim=True) / n
    sd = ((((yy - mu) * w) ** 2).sum(-1, keepdim=True) / n).sqrt().clamp_min(1e-8)
    ys = (yy - mu) / sd * w
    cobs = torch.isfinite(cov) & use[..., None]
    cc = torch.where(cobs, cov, 0).double()
    cn = cobs.sum(-1, keepdim=True).clamp_min(1)
    cm = cc.sum(-1, keepdim=True) / cn
    cs = ((torch.where(cobs, cc - cm, 0) ** 2).sum(-1, keepdim=True) / cn).sqrt()
    xs = torch.where(cobs & (cs > 0), (cc - cm) / cs.clamp_min(1e-8), 0)
    x = torch.cat([torch.ones_like(xs[:, :1]), xs], 1)  # [B,K,L]
    k = 1 + use.sum(-1, keepdim=True).double()
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    lam = 1.0 + 0.05 * n * k / 16 + 2.0 * k

    def fit_on(weight):
        xw = x * weight[:, None]
        gram = xw @ x.transpose(-1, -2)
        xty = (xw @ ys[..., None])[..., 0]
        return torch.linalg.solve(gram + lam[..., None] * eye, xty[..., None])[..., 0]

    beta = fit_on(w)
    whole = x
    if skip:  # the skipped steps: intercept 1, covariates 0
        whole = torch.cat(
            [F.pad(x[:, :1], (skip, 0), value=1.0), F.pad(x[:, 1:], (skip, 0))], 1
        )
    fit = torch.einsum("bk,bkl->bl", beta, whole)
    rank = w.cumsum(-1)
    early = w * (rank <= 0.8 * n)
    late = w - early
    pred = torch.einsum("bk,bkl->bl", fit_on(early), x)
    ln = late.sum(-1).clamp_min(1)
    err = (((ys - pred) * late) ** 2).sum(-1) / ln
    var = (((ys - (ys * late).sum(-1, keepdim=True) / ln[..., None]) * late) ** 2).sum(
        -1
    ) / ln
    r2 = 1 - err / var.clamp_min(1e-6)
    ok = (n[:, 0] >= 3 * k[:, 0] + 16) & (late.sum(-1) >= 8) & (r2 > 0) & use.any(-1)
    return (fit * sd + mu).float(), ok


def known_rows(cov: torch.Tensor, known: torch.Tensor, binary: bool, patch: int):
    """Known-future rows of cov [..., L] (known [...]): two-level rows as (x - min) / (max - min)
    when `binary`, then every row shifted one patch left. Returns (rows, two-level flags)."""
    if binary:
        finite = torch.isfinite(cov)
        lo = torch.where(finite, cov, float("inf")).amin(-1, keepdim=True)
        hi = torch.where(finite, cov, float("-inf")).amax(-1, keepdim=True)
        two = (
            ((cov == lo) | (cov == hi) | ~finite).all(-1)
            & known
            & (hi[..., 0] > lo[..., 0])
        )
        rows = torch.where(two[..., None], (cov - lo) / (hi - lo).clamp_min(1e-12), cov)
    else:
        two = torch.zeros_like(known)
        rows = cov
    return F.pad(rows[..., patch:], (0, patch), value=float("nan")), two


def varying(cov: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """[..., C] covariates (`finite`: their observed steps) with at least two observed values and
    a nonzero spread."""
    count = finite.sum(-1)
    mean = torch.where(finite, cov, 0).sum(-1) / count.clamp_min(1)
    spread = (torch.where(finite, cov - mean[..., None], 0) ** 2).sum(-1)
    return (count >= 2) & (spread > 0)


def covariate_rows(
    config,
    values: torch.Tensor,
    series_ids: torch.Tensor,
    covariates: torch.Tensor,
    fit_keep: torch.Tensor | None = None,
    skip: int = 0,
):
    """Append covariates [B,C,>=L] to values [B,V,L]; return values [B,V',L], ids/roles [B,V'].
    Return None if unusable; fit_keep [B] selects fits. Inputs omit `skip` all-missing steps.
    With `skip`, append the fit prefix [B,skip] (NaN if invalid), or None without a fit row."""
    patch = config.patch_length
    batch, variates, length = values.shape
    cov = covariates.to(device=values.device, dtype=values.dtype)
    if cov.shape[-1] < length:
        cov = F.pad(cov, (0, length - cov.shape[-1]), value=float("nan"))
    cov = cov[..., :length]
    position = torch.arange(length, device=values.device)
    last = torch.where(torch.isfinite(values), position, -1).amax(-1).amax(-1)
    finite = torch.isfinite(cov)
    valid = varying(cov, finite)
    known = valid & (finite & (position > last[:, None, None])).any(-1)  # [B,C]
    keep = known.any(0)
    if not bool(keep.any()):
        return None
    cov, known = cov[:, keep], known[:, keep]
    shifted, two = known_rows(cov, known, config.binary_rows, patch)
    roles = torch.where(two, ROLE_BINARY, ROLE_KNOWN)
    rows = torch.where(known[..., None], shifted, float("nan"))
    extra, extra_roles, extra_used = [rows], [roles], [known]
    head = None
    if config.fit_row and variates == 1:
        fit, ok = ridge_fit(values[:, 0], cov, known, skip=skip)
        if fit_keep is not None:
            ok = ok & fit_keep.to(ok.device)
        fit = torch.where(
            ok[:, None],
            F.pad(fit[..., patch:], (0, patch), value=float("nan")),
            float("nan"),
        )
        head, fit = fit[..., :skip], fit[..., skip:]
        extra.append(fit[:, None])
        extra_roles.append(torch.full_like(roles[:, :1], ROLE_FIT))
        extra_used.append(ok[:, None])
    rows = torch.cat(extra, 1)
    roles = torch.cat(extra_roles, 1)
    used = torch.cat(extra_used, 1)
    rows = torch.where(used[..., None], rows, float("nan"))
    # Callers' IDs become 0, 1, ...; negative IDs and zero role vectors isolate unused rows.
    k = rows.shape[1]
    series_ids = torch.unique(series_ids, return_inverse=True)[1]
    lone = (-1 - torch.arange(k, device=values.device))[None].expand(batch, k)
    ids = torch.where(used, series_ids[:, :1].expand(batch, k), lone)
    roles = torch.where(used, roles, ROLE_TARGET)
    target_roles = torch.zeros(batch, variates, dtype=roles.dtype, device=roles.device)
    grown = (
        torch.cat([values, rows], 1),
        torch.cat([series_ids, ids], 1),
        torch.cat([target_roles, roles], 1),
    )
    return (*grown, head) if skip else grown


def xreg_fit(
    values: torch.Tensor,
    covariates: torch.Tensor,
    future: torch.Tensor,
    gate: float = 0.05,
) -> torch.Tensor:
    """Return ridge contributions [B,V,L] without intercepts, in target units; zero without a fit.
    Inputs are values [B,V,L], covariates [B,C,L] and future flags [L]; `gate` is held-out R^2."""
    batch, _, length = values.shape
    if future.ndim == 1:
        future = future[None].expand(batch, length)
    obs = torch.isfinite(values) & ~future[:, None]
    w = obs.double()
    n = w.sum(-1)
    y = torch.where(obs, values, 0).double()
    mu = (y * w).sum(-1, keepdim=True) / n.clamp_min(1)[..., None]
    sd = (
        (((y - mu) * w) ** 2).sum(-1, keepdim=True) / n.clamp_min(1)[..., None]
    ).sqrt()
    sd = sd.clamp_min(1e-8)
    ys = (y - mu) / sd * w
    cobs = torch.isfinite(covariates)
    known = (cobs & future[:, None]).any(-1)
    cc = torch.where(cobs, covariates, 0).double()
    cn = cobs.sum(-1, keepdim=True).clamp_min(1)
    cm = cc.sum(-1, keepdim=True) / cn
    cs = ((torch.where(cobs, cc - cm, 0) ** 2).sum(-1, keepdim=True) / cn).sqrt()
    valid = known & (cs[..., 0] > 1e-12)
    xs = torch.where(cobs & valid[..., None], (cc - cm) / cs.clamp_min(1e-12), 0)
    x = torch.cat([torch.ones_like(xs[:, :1]), xs], 1)  # [B,K,L]
    k = valid.sum(-1).double() + 1
    lam = 1.0 + 0.05 * n * k[:, None] / 16 + 2.0 * k[:, None]
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

    def fit_on(weight):
        xw = x[:, None] * weight[:, :, None]
        gram = xw @ x[:, None].transpose(-1, -2)
        xty = (xw @ ys[..., None])[..., 0]
        return torch.linalg.solve(gram + lam[..., None, None] * eye, xty[..., None])[
            ..., 0
        ]

    beta = fit_on(w)
    rank = w.cumsum(-1)
    early = w * (rank <= 0.8 * n[..., None])
    late = w - early
    pred = torch.einsum("bvk,bkl->bvl", fit_on(early), x)
    lnc = late.sum(-1).clamp_min(1)
    err = (((ys - pred) * late) ** 2).sum(-1) / lnc
    lmean = (ys * late).sum(-1, keepdim=True) / lnc[..., None]
    var = (((ys - lmean) * late) ** 2).sum(-1) / lnc
    r2 = (1 - err / var.clamp_min(1e-6)).clamp(-1, 1)
    keep = (late.sum(-1) >= 8) & (r2 > gate) & (n >= 3 * k[:, None] + 16)
    keep = keep & valid.any(-1)[:, None]
    fitted = torch.einsum("bvk,bkl->bvl", beta[..., 1:], x[:, 1:]) * sd
    return torch.where(keep[..., None], fitted, 0.0).float()


def _missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return str(x) in ("<NA>", "NaT")


def encode_categorical(values, categories=None) -> np.ndarray:
    """Convert a 1-D column to float64 values or sorted category codes; missing/unknown -> NaN.
    A numeric column containing pandas NA/NaT is coded as categories; cast it to float
    before calling."""
    try:
        return np.asarray(values, dtype="float64")
    except (TypeError, ValueError):
        pass
    if categories is None:
        categories = sorted({str(x) for x in values if not _missing(x)})
    codes = {v: i for i, v in enumerate(categories)}
    return np.asarray(
        [np.nan if _missing(x) else codes.get(str(x), np.nan) for x in values],
        dtype="float64",
    )


def _known_future(column, length: int, shared: bool) -> np.ndarray:
    if shared:
        return encode_categorical(column)
    return np.concatenate(
        [encode_categorical(column[:length]), encode_categorical(column[length:])]
    )


def _columns(block) -> list:
    if block is None:
        return []
    if isinstance(block, np.ndarray) or torch.is_tensor(block):
        array = np.asarray(block)
        return [array] if array.ndim == 1 else list(array)
    if getattr(block, "ndim", None) == 1:
        return [np.asarray(block)]
    if hasattr(block, "items"):
        return [np.asarray(column) for _, column in block.items()]
    return [np.asarray(column) for column in block]


def prepare_covariates(
    length: int,
    horizon: int,
    future=None,
    past=None,
    max_covariates: int = MAX_COVARIATES,
    shared_categories: bool = True,
):
    """Return float32 covariates [C,length+horizon], or None; NaN marks missing values.
    `future` spans history and horizon; `past` spans history; columns align by position.
    Columns without two distinct values are dropped, and at most `max_covariates` are kept."""
    future, past = _columns(future), _columns(past)
    if any(column.ndim != 1 for column in future + past):
        raise ValueError(
            "each covariate must be a 1-D column: per series, pass a 1-D array or "
            "pandas Series (one covariate), a 2-D array [C, length], a list of 1-D "
            "arrays, a dict of name -> 1-D array or a pandas DataFrame"
        )
    rows = []
    for column in future:
        column = np.asarray(column)
        if len(column) != length + horizon:
            raise ValueError(
                f"a known-future covariate needs {length + horizon} values "
                f"(history {length} + horizon {horizon}), got {len(column)}"
            )
        rows.append(_known_future(column, length, shared_categories))
    for column in past:
        column = np.asarray(column)
        if len(column) != length:
            raise ValueError(
                f"a past covariate needs {length} values, got {len(column)}"
            )
        rows.append(
            np.concatenate([encode_categorical(column), np.full(horizon, np.nan)])
        )
    if not rows:
        return None
    covs = np.stack(rows)
    finite = np.isfinite(covs)
    with np.errstate(all="ignore"):
        spread = np.nanstd(np.where(finite, covs, np.nan), 1) if finite.any() else 0
    # Check variation in forecast precision before selecting columns.
    single = covs.astype("float32")
    low = np.where(finite, single, np.inf).min(1)
    high = np.where(finite, single, -np.inf).max(1)
    covs = single[(high > low) & (np.nan_to_num(spread) > 0)][:max_covariates]
    return covs if len(covs) else None


def usable_covariates(history: torch.Tensor, covariates: torch.Tensor, horizon: int):
    """Keep usable covariates [B,C,T+horizon] for history [B,V,T]; return None when empty."""
    t = history.shape[-1]
    cov = covariates.clone()
    finite = torch.isfinite(cov)
    position = torch.arange(t, device=history.device)
    seen = torch.isfinite(history).any(1)  # [B,T]
    last = torch.where(seen, position, -1).amax(-1)  # [B]
    hist = finite[..., :t]
    late = hist & (position > last[:, None])[:, None]
    keep = finite[..., t:].any(-1) | late.any(-1)
    if t > horizon:
        cut = position < t - horizon
        last_b = torch.where(seen & cut, position, -1).amax(-1)
        in_block = cut & (position > last_b[:, None])
        keep |= (hist & in_block[:, None]).any(-1)
    cov = torch.where(keep[..., None], cov, float("nan"))
    finite = torch.isfinite(cov)
    # Compare values exactly; floating-point variance can be nonzero for a constant.
    low = torch.where(finite, cov, float("inf")).amin(-1)
    high = torch.where(finite, cov, float("-inf")).amax(-1)
    cov = torch.where((high == low)[..., None], float("nan"), cov)
    finite = torch.isfinite(cov)
    usable = varying(cov, finite)
    beyond = usable & (usable.cumsum(-1) > MAX_COVARIATES)
    cov = torch.where(beyond[..., None], float("nan"), cov)
    columns = torch.isfinite(cov).any(-1).any(0)
    if not bool(columns.any()):
        return None
    return cov[:, columns]
