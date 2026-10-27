"""Attention masks and time positions for Decoupled Horizon Forecasting (DHF) decoding.

The sequence is `[Enc ; Dec]`: encoder tokens embed the observed context patches;
decoder tokens are forecast queries for the horizon patches. All masks are boolean,
True meaning "may attend".
"""

import torch
from torch import Tensor


def compute_enc_key_validity(obs_per_patch: Tensor, attend_missing: bool) -> Tensor:
    """`(R, N)` encoder key validity.

    A patch is a valid key if it is observed, or if `attend_missing` is set and the
    row's series has started at or before it. Leading all-unobserved patches are
    left-padding and are never valid keys.
    """
    if not attend_missing:
        return obs_per_patch
    started = obs_per_patch.cummax(dim=-1).values
    return obs_per_patch | started


def enc_causal_mask(key_valid: Tensor) -> Tensor:
    """`(R, L, L)` causal encoder mask over valid keys, with the diagonal always open."""
    _, ll = key_valid.shape
    device = key_valid.device
    causal = torch.tril(torch.ones(ll, ll, dtype=torch.bool, device=device))
    eye = torch.eye(ll, dtype=torch.bool, device=device)
    return (causal[None] & key_valid[:, None, :]) | eye[None]


def build_forecast_time_mask(context_len: int, horizon: int, key_valid: Tensor, device: torch.device) -> Tensor:
    """`(R, 1, L+h, L+h)` time mask for one block without covariates.

    The encoder is causal; every decoder token reads the whole valid context and its
    own causal decoder chain; the context never reads the horizon.
    """
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = torch.zeros(rows, context_len, horizon, dtype=torch.bool, device=device)
    dec_enc = key_valid[:, None, :].expand(rows, horizon, context_len)
    dec_dec = torch.tril(torch.ones(horizon, horizon, dtype=torch.bool, device=device))[None].expand(rows, horizon, horizon)
    top = torch.cat([enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def build_forecast_positions(context_len: int, horizon: int, first_obs: Tensor) -> Tensor:
    """`(R, L+h)` time positions `0..L+h-1`, anchored at each row's first observed patch."""
    p = torch.arange(context_len + horizon, device=first_obs.device)
    return p[None, :] - first_obs[:, None]


def compute_block_layout(min_context: Tensor, n_patches: int, block_size: int) -> tuple[Tensor, Tensor]:
    """Per decoder position: the anchor of its block and whether it is active (`p >= s`)."""
    device = min_context.device
    p = torch.arange(n_patches, device=device)[None, :]
    s = min_context[:, None]
    active = p >= s
    offset = (p - s).clamp(min=0)
    anchor = s + block_size * (offset // block_size)
    anchor = torch.where(active, anchor, torch.zeros_like(anchor))
    return anchor, active


def build_two_stream_time_mask(
    min_context: Tensor,
    n_patches: int,
    block_size: int,
    key_valid: Tensor,
    bidirectional_rows: Tensor | None,
) -> Tensor:
    """`(R, 1, 2N, 2N)` time mask over a full `[Enc ; Dec]` window (the covariate path).

    - Enc to Enc: causal over valid keys; full (non-causal) for `bidirectional_rows`,
      the known-future covariates.
    - Enc to Dec: never.
    - Dec to Enc: a decoder token in the block anchored at `b` reads valid `Enc[j]`, `j < b`.
    - Dec to Dec: causal within the same block; every decoder token reads itself.
    """
    anchor, active = compute_block_layout(min_context, n_patches, block_size)
    rows, n = active.shape
    device = active.device
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    key_col = key_valid[:, None, :]
    j = torch.arange(n, device=device).view(1, 1, n)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = key_valid[:, None, :].expand(rows, n, n) | eye[None]
        enc_enc = torch.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = torch.zeros(rows, n, n, dtype=torch.bool, device=device)
    dec_enc = active[:, :, None] & (j < anchor[:, :, None]) & key_col
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = same_block & causal[None] & active[:, :, None] & active[:, None, :]
    dec_dec = dec_dec | eye[None]
    top = torch.cat([enc_enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def variate_firewall(r: int, device: torch.device, target_rows: Tensor | None, bidirectional_rows: Tensor | None) -> Tensor | None:
    """`(R, R)` forbidden query-to-key row pairs, or None.

    - A covariate row may not read a target row (covariates are exogenous).
    - A known-future (bidirectional) row may not read a causal row. Without this, a
      covariate's view of the future could relay a target's own future back into it.
    """
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = torch.zeros(r, r, dtype=torch.bool, device=device)
    if target_rows is not None:
        forbid = forbid | ((~target_rows[:, None]) & target_rows[None, :])
    if bidirectional_rows is not None:
        forbid = forbid | (bidirectional_rows[:, None] & (~bidirectional_rows)[None, :])
    return forbid


def build_two_stream_variate_mask(
    group_ids: Tensor,
    enc_key_valid: Tensor,
    n_dec: int,
    target_rows: Tensor | None = None,
    bidirectional_rows: Tensor | None = None,
) -> Tensor | None:
    """`(N_enc + N_dec, 1, R, R)` variate mask, or None for a single row.

    A row reads another row of the same series; on encoder slots the key row must also
    be a valid key. The firewalls apply on every slot; the diagonal is always open.
    """
    r, _ = enc_key_valid.shape
    if r == 1:
        return None
    device = enc_key_valid.device
    row_eq = group_ids[:, None] == group_ids[None, :]
    dec_valid = torch.ones(r, n_dec, dtype=torch.bool, device=device)
    slot_valid = torch.cat([enc_key_valid, dec_valid], dim=1)
    key_valid = slot_valid.permute(1, 0)
    mask = row_eq[None, None] & key_valid[:, None, None, :]
    forbid = variate_firewall(r, device, target_rows, bidirectional_rows)
    if forbid is not None:
        mask = mask & ~forbid[None, None]
    eye = torch.eye(r, dtype=torch.bool, device=device)
    return mask | eye[None, None]


def build_two_stream_positions(n_patches: int, first_obs: Tensor) -> Tensor:
    """`(R, 2N)` positions: encoder and decoder token of the same patch share a time."""
    p = torch.arange(n_patches, device=first_obs.device)
    row = p[None, :] - first_obs[:, None]
    return torch.cat([row, row], dim=-1)


def group_first_observed_patch(obs_per_patch: Tensor, inverse: Tensor, num_groups: int) -> Tensor:
    """`(num_groups,)` index of the first patch any row of the group observes (0 if none)."""
    n = obs_per_patch.shape[-1]
    idx = inverse[:, None].expand(-1, n)
    group_obs = torch.zeros(num_groups, n, dtype=torch.float32, device=obs_per_patch.device)
    group_obs.scatter_reduce_(0, idx, obs_per_patch.float(), reduce="amax", include_self=True)
    return group_obs.argmax(dim=-1)
