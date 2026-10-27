"""The Timestral network and its single-block forecast primitive.

The model normalizes each row causally per patch, embeds patches from `[values,
missing]` channels, runs a time/variate transformer over `[Enc ; Dec]` tokens (context
patches followed by learned forecast-query tokens), and reads quantiles for every
horizon time step from the decoder tokens.
"""

import torch
from torch import Tensor, nn

from sktime.libs.timestral.config import TimestralConfig
from sktime.libs.timestral.layers import ResidualBlock, Transformer, TwoStreamMask
from sktime.libs.timestral.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from sktime.libs.timestral.scaler import (
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
    instance_norm_stats,
)

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


class PatchRepresentation(nn.Module):
    """Patch embedding over per-step `[values, missing]` channels, plus the forecast-query token."""

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualBlock(2 * patch_size, 4 * d_model, d_model, use_bias=True)
        self.forecast_query = nn.Parameter(torch.zeros(d_model))

    def forward(self, scaled: Tensor, observed: Tensor) -> Tensor:
        """`(R, T)` normalized values and observation mask to `(R, T / patch_size, d_model)`."""
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).to(scaled.dtype)
        x = torch.cat([values, missing], dim=-1).to(self.patch_proj.hidden_layer.weight.dtype)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int) -> Tensor:
        return self.forecast_query.view(1, 1, -1).expand(rows, n_patches, -1)


def monotone_quantiles(raw: Tensor, levels: tuple[float, ...], width_factor: float) -> Tensor:
    """Non-crossing quantiles from raw head outputs along the last axis (`monotone_cumsum`).

    The median channel is free. The others fan out from it as
    `median -/+ cumsum(width_factor * dp * softplus(raw))`, `dp` being the probability gap
    to the next level inward, so unequal knot spacing is respected.
    """
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    widths_t = torch.tensor(widths, device=raw.device, dtype=raw.dtype)
    inc = width_factor * widths_t * nn.functional.softplus(raw)
    median = raw[..., m : m + 1]
    up = torch.cumsum(inc[..., m + 1 :], dim=-1)
    down = torch.flip(torch.cumsum(torch.flip(inc[..., :m], dims=[-1]), dim=-1), dims=[-1])
    big = torch.finfo(raw.dtype).max
    return torch.cat([median - down, median, median + up], dim=-1).clamp(min=-big, max=big)


class QuantileHead(nn.Module):
    """Maps each decoder token to `patch_size` monotone quantile vectors in normalized space."""

    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.projection = ResidualBlock(
            config.d_model, 4 * config.d_model, config.patch_size * config.num_quantiles, use_bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """`(R, N, D)` to normalized quantiles `(R, N * patch_size, Q)` in at least fp32."""
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(*leading, n_patches, self.patch_size, self.num_quantiles)
        preds = preds.to(torch.promote_types(preds.dtype, torch.float32))
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return preds.reshape(*leading, n_patches * self.patch_size, self.num_quantiles)


class TimestralModel(nn.Module):
    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.representation = PatchRepresentation(c.patch_size, c.d_model)
        self.transformer = Transformer(
            block_schedule=c.block_schedule,
            embed_dim=c.d_model,
            num_heads=c.num_heads,
            mlp_hidden_dim=c.mlp_hidden_dim,
            rope_theta=c.rope_theta,
            rope_dim=c.rope_dim,
            norm_eps=c.norm_eps,
            attn_bias=c.attn_bias,
            mlp_bias=c.mlp_bias,
            time_attn_gate=c.time_attn_gate,
            variate_attn_gate=c.variate_attn_gate,
        )
        self.head = QuantileHead(c)

    def scale(self, inputs: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Causal per-patch normalization of packed `(R, T)` rows.

        Returns `(scaled, loc, scale)`, each `(R, T)`.
        """
        c = self.config
        loc, scale = causal_patch_loc_scale(
            inputs.unsqueeze(1), observed.unsqueeze(1), c.patch_size, c.minimum_scale, c.stabilize_with_global
        )
        loc, scale = loc.squeeze(1), scale.squeeze(1)
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh)
        return scaled, loc, scale

    def _known_cov_scaling(
        self, inputs: Tensor, observed: Tensor, scaled: Tensor, loc: Tensor, scale: Tensor, known_rows: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize known-future covariate rows with non-causal whole-row statistics.

        Indicator rows (values in {0, 1}) skip the asinh squash.
        """
        if not bool(known_rows.any()):
            return scaled, loc, scale
        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, minimum_scale=self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            work = torch.promote_types(known_scaled.dtype, torch.float32)
            squashed = known_scaled.to(work).asinh().to(known_scaled.dtype)
            known_scaled = torch.where(observed & ~is_binary, squashed, known_scaled)
        sel = known_rows[:, None]
        return torch.where(sel, known_scaled, scaled), torch.where(sel, loc_c, loc), torch.where(sel, scale_c, scale)

    def _run_two_stream(
        self, x_enc: Tensor, x_dec: Tensor, time_mask: Tensor, variate_mask: Tensor | None, positions: Tensor
    ) -> Tensor:
        """Transformer over `[Enc ; Dec]`, then the head on the decoder tokens (normalized space)."""
        x = torch.cat([x_enc, x_dec], dim=1)
        mask: Tensor | TwoStreamMask = time_mask
        if self.config.split_context_horizon_attn:
            mask = TwoStreamMask(mask=time_mask, enc_len=x_enc.shape[1])
        h = self.transformer(x, mask, variate_mask, positions)
        return self.head(h[:, x_enc.shape[1] :, :])

    def predict_block(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        role_ids: Tensor | None = None,
        known_future: Tensor | None = None,
        known_future_mask: Tensor | None = None,
    ) -> Tensor:
        """Forecast one block of `horizon_patches` patches after a packed context.

        Args:
            inputs: `(R, T)` context values, `T` a multiple of the patch size.
            observed: `(R, T)` observation mask.
            group_ids: `(R,)` series id per row; rows of one series attend each other.
            horizon_patches: Number of patches to forecast.
            role_ids: `(R,)` row roles (`ROLE_TARGET`, `ROLE_PAST_COV`, `ROLE_KNOWN_COV`),
                or None when every row is a target.
            known_future: `(R, horizon_patches * patch_size)` known-future covariate
                values over this block; zeros on other rows.
            known_future_mask: Observation mask for `known_future`.

        Returns:
            Raw-space quantiles `(R, horizon_patches * patch_size, Q)`, non-decreasing along Q.
        """
        assert inputs.dim() == 2, f"expected packed (R, T) inputs, got {tuple(inputs.shape)}"
        if role_ids is None or not self.config.covariate_enabled or bool((role_ids == ROLE_TARGET).all()):
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)
        r = inputs.shape[0]
        horizon_len = horizon_patches * self.config.patch_size
        if known_future is None:
            known_future = torch.zeros(r, horizon_len, device=inputs.device, dtype=inputs.dtype)
            known_future_mask = torch.zeros(r, horizon_len, device=inputs.device, dtype=torch.bool)
        assert known_future_mask is not None, "known_future_mask is required alongside known_future"
        assert known_future.shape == (r, horizon_len), f"known_future must be ({r}, {horizon_len}), got {tuple(known_future.shape)}"
        full_inputs = torch.cat([inputs, known_future], dim=-1)
        full_observed = torch.cat([observed, known_future_mask], dim=-1)
        return self._predict_window(full_inputs, full_observed, group_ids, horizon_patches, role_ids)

    def _predict_context_only(self, inputs: Tensor, observed: Tensor, group_ids: Tensor, horizon_patches: int) -> Tensor:
        """No covariates: encoder over the context, `h` forecast-query tokens after it."""
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        assert t % ps == 0, f"context length {t} must be a multiple of patch_size {ps}"
        assert group_ids.shape == (rows,), f"group_ids must be ({rows},), got {tuple(group_ids.shape)}"
        ll, h = t // ps, horizon_patches
        assert h >= 1, f"horizon_patches must be >= 1, got {h}"

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid, inputs.device)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(preds, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), c.use_asinh)

    def _predict_window(
        self, full_inputs: Tensor, full_observed: Tensor, group_ids: Tensor, horizon_patches: int, role_ids: Tensor
    ) -> Tensor:
        """With covariates: encoder over context plus horizon, so known-future values are keys.

        Target and past-only rows keep a blank horizon. The time and variate masks keep
        target information out of covariate rows and keep any row's future out of the
        targets.
        """
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        assert w % ps == 0, f"window length {w} must be a multiple of patch_size {ps}"
        assert group_ids.shape == (rows,) and role_ids.shape == (rows,), "group_ids and role_ids must be (R,)"
        h = horizon_patches
        assert 1 <= h <= c.block_size, f"decode block {h} must be in [1, {c.block_size}] with covariates"
        n = w // ps
        ll = n - h
        assert ll >= 1, f"context must have at least one patch (window {n}, horizon {h})"
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.unflatten(-1, (n, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        s_rows = torch.full((rows,), ll, device=full_inputs.device, dtype=torch.long)
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_two_stream_positions(n, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions).unflatten(1, (n, ps))
        horizon_preds = preds[:, ll:n].reshape(rows, h * ps, preds.shape[-1])
        anchor_loc = loc.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(horizon_preds, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), c.use_asinh)
