"""Probabilistic forecasting with Timestral, including the clampsym test-time augmentation.

`TimestralForecaster.forecast` takes a context `(B, V, T)` (series, variates, time; NaN
marks a missing value) and returns quantile forecasts `(B, V, H, Q)`. Horizons longer
than one decode block are forecast block by block, feeding each block's median back as
context.

Clampsym (both options on by default):

- Symmetric averaging forecasts the context and its negation, reflects the second
  forecast back (`-Q(1 - p)`), and averages the two. Every covariate row is negated
  with the context: numeric covariates, 0/1 indicators, target encodings and raw codes.
- Non-negative clipping clamps the forecast at 0 for every series whose observed
  context is entirely non-negative.

Categorical covariates are target-encoded against the series' forward-filled context
(see :mod:`timestral.covariate_encoding`) and then handled like numeric covariates.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sktime.libs.timestral.covariate_encoding import target_encode
from sktime.libs.timestral.loading import load_model
from sktime.libs.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET, TimestralModel
from sktime.libs.timestral.scaler import advancing_block_frames, causal_patch_loc_scale, clamp_nonfinite

ArrayLike = Tensor | np.ndarray


@dataclass(frozen=True)
class Forecast:
    """Quantile forecast.

    Attributes:
        quantiles: `(B, V, H, Q)` forecast values, sorted along the quantile axis.
        quantile_levels: The `Q` probability levels.
    """

    quantiles: Tensor
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> Tensor:
        """`(B, V, H)` median forecast."""
        return self.quantiles[..., self.quantile_levels.index(0.5)]

    def quantile(self, level: float) -> Tensor:
        """`(B, V, H)` forecast at one of the model's quantile levels."""
        if level not in self.quantile_levels:
            raise ValueError(f"level {level} is not one of {self.quantile_levels}")
        return self.quantiles[..., self.quantile_levels.index(level)]


def ffill_with_mask(inputs: Tensor, mask: Tensor) -> Tensor:
    """Forward-fill unobserved positions; leading unobserved positions become 0."""
    if mask.all():
        return inputs
    idx = torch.arange(inputs.shape[-1], device=inputs.device).expand_as(inputs)
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    idx = idx.cummax(dim=-1).values
    return torch.nan_to_num(torch.gather(inputs, dim=-1, index=idx), nan=0.0)


def pad_to_patch(inputs: Tensor, mask: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Left-pad to a multiple of `patch_size`; padding is 0 and unobserved."""
    ctx_len = inputs.shape[-1]
    if ctx_len % patch_size != 0:
        pad_len = patch_size - (ctx_len % patch_size)
        inputs = torch.nn.functional.pad(inputs, (pad_len, 0), value=0.0)
        mask = torch.nn.functional.pad(mask, (pad_len, 0), value=False)
    return inputs, mask


def replace_extreme_values(x: Tensor, threshold: float, replacement: float = 0.0) -> Tensor:
    return torch.where(torch.isfinite(x) & (x.abs() < threshold), x, torch.full_like(x, replacement))


def nonnegative_context(inputs: Tensor, observation_mask: Tensor) -> Tensor:
    """`(B, V)`: whether every observed context value is non-negative (and one exists)."""
    observed = torch.isfinite(inputs) & observation_mask.bool()
    return observed.any(dim=-1) & ((inputs >= 0) | ~observed).all(dim=-1)


class TimestralForecaster:
    """Block-autoregressive quantile forecaster around a :class:`TimestralModel`.

    Args:
        model: The network.
        max_context: Context cap in time steps (most recent kept); defaults to the
            trained context length.
        per_shot_horizon_patches: Patches decoded per block, capped at the trained
            block size.
        quantile_real_cap_k: Clamp forecasts to `[min - k * s, max + k * s]` of the
            observed context, `s` being the scale of the last context patch. 0 disables.
        extreme_value_threshold: Fed-back medians at or above this magnitude become 0.
    """

    def __init__(
        self,
        model: TimestralModel,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        extreme_value_threshold: float = 1e30,
    ) -> None:
        assert per_shot_horizon_patches >= 1, f"per_shot_horizon_patches must be >= 1, got {per_shot_horizon_patches}"
        self.model = model
        self.config = model.config
        self.max_context = max_context if max_context is not None else model.config.context_length
        self.per_shot_horizon_patches = min(per_shot_horizon_patches, model.config.block_size)
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        self.quantile_levels = tuple(model.config.quantile_knots)
        self.median_index = model.config.median_index

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        device: str | torch.device = "cpu",
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> "TimestralForecaster":
        """Load an exported checkpoint (`config.json` + `model.safetensors`).

        `path_or_repo_id` is a local directory or a Hugging Face Hub repo id. `revision`,
        `cache_dir`, `token` and `local_files_only` only apply to Hub repos.
        """
        model = load_model(path_or_repo_id, device=device, revision=revision, cache_dir=cache_dir, token=token,
                           local_files_only=local_files_only)
        return cls(model, max_context, per_shot_horizon_patches, quantile_real_cap_k)

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def supports_covariates(self) -> bool:
        return self.config.covariate_enabled

    @property
    def supports_known_covariates(self) -> bool:
        return self.config.supports_known_covariates

    @torch.inference_mode()
    def forecast(
        self,
        context: ArrayLike,
        prediction_length: int,
        observation_mask: ArrayLike | None = None,
        past_covariates: ArrayLike | None = None,
        known_covariates: ArrayLike | None = None,
        categorical_past_covariates: ArrayLike | None = None,
        categorical_known_covariates: ArrayLike | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> Forecast:
        """Forecast `prediction_length` steps after `context`.

        Args:
            context: `(B, V, T)`, `(B, T)` (one variate) or `(T,)` (one series). NaN marks
                a missing value. The variates of a series are forecast jointly.
            prediction_length: Horizon in time steps.
            observation_mask: Optional boolean mask shaped like `context`; False marks
                additional missing values.
            past_covariates: `(B, C_past, T)` covariates known over the context only.
            known_covariates: `(B, C_known, T + prediction_length)` covariates known over
                the context and the horizon.
            categorical_past_covariates: `(B, C, T)` category codes (any numbers; NaN marks
                a missing value) known over the context only.
            categorical_known_covariates: `(B, C, T + prediction_length)` category codes
                known over the context and the horizon.
                Categorical covariates are target-encoded per series against the
                forward-filled target context. A series with several target variates
                has no single target to encode against, so its codes are used as is.
            symmetric_averaging: Average with the reflected forecast of `-context`, with every
                covariate row negated too.
            clip_nonnegative: Clamp at 0 the series whose observed context is non-negative.

        Returns:
            A :class:`Forecast` with `quantiles` `(B, V, prediction_length, Q)`.
        """
        assert prediction_length >= 1, f"prediction_length must be >= 1, got {prediction_length}"
        inputs, mask = self._as_context(context, observation_mask)
        b = inputs.shape[0]
        t = inputs.shape[-1]
        past = self._as_covariates(past_covariates, b, t, "past_covariates")
        known = self._as_covariates(known_covariates, b, t + prediction_length, "known_covariates")
        cat_past = self._as_categorical(categorical_past_covariates, b, t, "categorical_past_covariates")
        cat_known = self._as_categorical(
            categorical_known_covariates, b, t + prediction_length, "categorical_known_covariates"
        )
        covariates = {"past_covariates": past, "known_covariates": known}
        if cat_past is not None or cat_known is not None:
            target = self._encoding_target(inputs, mask)
            enc_past = self._encode_categorical(cat_past, target)
            enc_known = self._encode_categorical(cat_known, target)
            covariates = {"past_covariates": _cat_rows(past, enc_past), "known_covariates": _cat_rows(known, enc_known)}
        # The mirrored pass forecasts -context with every covariate row negated. Target encodings
        # are linear in the target, so negating them gives the encodings against -context.
        mirrored_covariates = {name: _negate(rows) for name, rows in covariates.items()}
        if covariates["known_covariates"] is not None and not self.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")

        quantiles = self._forecast_raw(inputs, mask, prediction_length, **covariates)
        if symmetric_averaging:
            levels = self.quantile_levels
            assert all(abs(lo + hi - 1.0) < 1e-9 for lo, hi in zip(levels, reversed(levels))), (
                f"symmetric averaging needs quantile levels symmetric about 0.5, got {levels}"
            )
            mirrored = self._forecast_raw(-inputs, mask, prediction_length, **mirrored_covariates)
            quantiles = 0.5 * (quantiles + (-mirrored).flip(-1))
        if clip_nonnegative:
            eligible = nonnegative_context(inputs, mask)
            quantiles = torch.where(eligible[..., None, None], quantiles.clamp_min(0), quantiles)
        return Forecast(quantiles=quantiles, quantile_levels=self.quantile_levels)

    def _as_context(self, context: ArrayLike, observation_mask: ArrayLike | None) -> tuple[Tensor, Tensor]:
        x = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x[None, None]
        elif x.dim() == 2:
            x = x[:, None]
        assert x.dim() == 3, f"context must be (T,), (B, T) or (B, V, T), got {tuple(x.shape)}"
        mask = torch.isfinite(x)
        if observation_mask is not None:
            user_mask = torch.as_tensor(observation_mask, dtype=torch.bool, device=self.device).reshape(x.shape)
            mask = mask & user_mask
        return torch.where(mask, x, torch.zeros_like(x)), mask

    def _encoding_target(self, inputs: Tensor, mask: Tensor) -> Tensor | None:
        """`(B, T)` target to encode against, or None when a series has several target variates.

        The forward-filled target from its first observation on (so left padding or a late
        start does not enter the category means), over the last `max_context` steps, the
        ones the model reads; NaN elsewhere.
        """
        if inputs.shape[1] != 1:
            return None
        t = inputs.shape[-1]
        started = mask[:, 0].int().cummax(dim=-1).values.bool()
        in_window = torch.arange(t, device=inputs.device) >= t - self.max_context
        return torch.where(started & in_window, ffill_with_mask(inputs, mask)[:, 0], torch.nan)

    def _encode_categorical(self, codes: Tensor | None, target_context: Tensor | None) -> Tensor | None:
        """Target-encode `(B, C, L)` codes against `(B, T)` targets (NaN outside the encoding window); raw codes when None."""
        if codes is None or target_context is None:
            return codes.to(self.device, torch.float32) if codes is not None else None
        t = target_context.shape[-1]
        ctx = target_context.double().cpu().numpy()
        rows = codes.double().cpu().numpy()
        out = np.empty(rows.shape, dtype=np.float32)
        for i in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                future = rows[i, j, t:] if rows.shape[-1] > t else None
                past_enc, future_enc = target_encode(ctx[i], rows[i, j, :t], future)
                # A missing code before the target starts is padding, not the missing category.
                out[i, j, :t] = np.where(np.isnan(ctx[i]) & np.isnan(rows[i, j, :t]), np.nan, past_enc)
                if future_enc is not None:
                    out[i, j, t:] = future_enc
        return torch.from_numpy(out).to(self.device)

    def _as_categorical(self, values: ArrayLike | None, batch: int, length: int, name: str) -> Tensor | None:
        """Category codes as float64 on CPU, so that distinct codes past 2**24 stay distinct."""
        if values is None:
            return None
        v = self._as_covariates(torch.as_tensor(values, dtype=torch.float64), batch, length, name, dtype=torch.float64)
        if v is not None and torch.isinf(v).any():
            raise ValueError(f"{name} must hold finite codes or NaN; got an infinite value")
        return v.cpu() if v is not None else None

    def _as_covariates(
        self, values: ArrayLike | None, batch: int, length: int, name: str, dtype: torch.dtype = torch.float32
    ) -> Tensor | None:
        if values is None:
            return None
        if not self.supports_covariates:
            raise ValueError(f"this checkpoint was not trained with covariates; got {name}")
        v = torch.as_tensor(values, dtype=dtype, device=self.device if dtype == torch.float32 else "cpu")
        if v.dim() == 2:
            v = v[:, None]
        assert v.dim() == 3 and v.shape[0] == batch and v.shape[-1] == length, (
            f"{name} must be (B={batch}, C, {length}), got {tuple(v.shape)}"
        )
        return v if v.shape[1] > 0 else None

    def _forecast_raw(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        past_covariates: Tensor | None = None,
        known_covariates: Tensor | None = None,
    ) -> Tensor:
        """One forecast pass: `(B, V, T)` inputs to `(B, V, H, Q)` raw-space quantiles."""
        self.model.eval()
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = self._assemble_rows(
            inputs, observation_mask, prediction_length, past_covariates, known_covariates
        )
        quantiles = self._block_ar_decode(rows, mask, prediction_length, role_ids, known_future, known_future_mask)
        quantiles = quantiles[:, :n_tgt]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(quantiles, rows[:, :n_tgt], mask[:, :n_tgt])
        if torch.isnan(quantiles).any():
            raise ValueError(f"forecast produced {int(torch.isnan(quantiles).sum())} NaN values; the input is degenerate")
        return quantiles

    def _cap_quantiles(self, quantiles: Tensor, inputs: Tensor, observation_mask: Tensor) -> Tensor:
        not_obs = ~observation_mask
        ctx_max = torch.nan_to_num(inputs.masked_fill(not_obs, float("-inf")).amax(-1, keepdim=True), neginf=0.0)
        ctx_min = torch.nan_to_num(inputs.masked_fill(not_obs, float("inf")).amin(-1, keepdim=True), posinf=0.0)
        c = self.config
        _, scale = causal_patch_loc_scale(inputs, observation_mask, c.patch_size, c.minimum_scale, c.stabilize_with_global)
        anchor_scale = scale[..., -1:] * self.quantile_real_cap_k
        return quantiles.clamp((ctx_min - anchor_scale).unsqueeze(-1), (ctx_max + anchor_scale).unsqueeze(-1))

    def _assemble_rows(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        past_covariates: Tensor | None,
        known_covariates: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, int]:
        """Stack target and covariate rows; returns `(rows, mask, role_ids, known_future, known_future_mask, n_tgt)`."""
        ps = self.patch_size
        device = inputs.device
        b, n_tgt, ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps

        ctx_rows = [ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt

        n_past = 0
        if past_covariates is not None:
            assert past_covariates.shape[-1] == ctx0, "past covariates must cover exactly the context"
            n_past = past_covariates.shape[1]
            ctx_masks.append(~torch.isnan(past_covariates))
            ctx_rows.append(torch.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past

        n_known = 0
        known_future_vals: Tensor | None = None
        known_future_obs: Tensor | None = None
        if known_covariates is not None:
            n_known = known_covariates.shape[1]
            kc_ctx, kc_fut = known_covariates[..., :-prediction_length], known_covariates[..., -prediction_length:]
            assert kc_ctx.shape[-1] == ctx0, "known covariates must cover the context plus the horizon"
            ctx_masks.append(~torch.isnan(kc_ctx))
            ctx_rows.append(torch.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = ~torch.isnan(kc_fut)
            known_future_vals = torch.nan_to_num(kc_fut, nan=0.0)

        rows = torch.cat(ctx_rows, dim=1)
        mask = torch.cat(ctx_masks, dim=1)
        if rows.shape[-1] > self.max_context:
            rows = rows[:, :, -self.max_context :]
            mask = mask[:, :, -self.max_context :]
        rows, mask = pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return rows, mask, None, None, None, n_tgt

        role_ids = torch.tensor(roles, dtype=torch.long, device=device).repeat(b)
        known_future: Tensor | None = None
        known_future_mask: Tensor | None = None
        if n_known > 0:
            assert known_future_vals is not None and known_future_obs is not None
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = torch.zeros(b, v_all, horizon, device=device, dtype=rows.dtype)
            kfm = torch.zeros(b, v_all, horizon, device=device, dtype=torch.bool)
            kf[:, k0 : k0 + n_known, :plen] = known_future_vals
            kfm[:, k0 : k0 + n_known, :plen] = known_future_obs
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)
        return rows, mask, role_ids, known_future, known_future_mask, n_tgt

    def _block_ar_decode(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        role_ids: Tensor | None,
        known_future: Tensor | None,
        known_future_mask: Tensor | None,
    ) -> Tensor:
        """Decode block by block in the packed `(R, T)` layout; returns `(B, V, H, Q)`.

        After each block the context grows by the block's feedback: targets carry their
        median, known-future covariates their real future, past-only covariates a blank.
        Past `max_context` the oldest steps are dropped.
        """
        ps = self.patch_size
        b, v, ctx_len = inputs.shape
        context = inputs.reshape(b * v, ctx_len)
        mask = observation_mask.reshape(b * v, ctx_len)
        group_ids = torch.arange(b, device=inputs.device).repeat_interleave(v)

        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET

        blocks: list[Tensor] = []
        remaining = math.ceil(prediction_length / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_shot_horizon_patches, remaining)
            blen = block * ps
            kf_block = known_future[:, offset : offset + blen] if known_future is not None else None
            kfm_block = known_future_mask[:, offset : offset + blen] if known_future_mask is not None else None
            quantiles = self.model.predict_block(context, mask, group_ids, block, role_ids, kf_block, kfm_block)
            quantiles = self._refine_block(quantiles, context, mask, target_rows)
            blocks.append(quantiles)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break
            feedback = replace_extreme_values(quantiles[..., self.median_index], self.extreme_value_threshold)
            feedback_mask = torch.ones_like(feedback, dtype=torch.bool)
            if role_ids is not None:
                assert known_rows is not None and past_rows is not None
                if kf_block is not None:
                    assert kfm_block is not None
                    feedback = torch.where(known_rows[:, None], kf_block, feedback)
                    feedback_mask = torch.where(known_rows[:, None], kfm_block, feedback_mask)
                feedback = torch.where(past_rows[:, None], torch.zeros_like(feedback), feedback)
                feedback_mask = feedback_mask & ~past_rows[:, None]
            context = torch.cat([context, feedback], dim=-1)
            mask = torch.cat([mask, feedback_mask], dim=-1)
            if context.shape[-1] > self.max_context:
                keep = max(ps, (self.max_context // ps) * ps)
                context = context[..., -keep:]
                mask = mask[..., -keep:]

        packed = torch.cat(blocks, dim=-2)[..., :prediction_length, :]
        return packed.reshape(b, v, prediction_length, -1)

    def _refine_block(self, quantiles: Tensor, context: Tensor, mask: Tensor, target_rows: Tensor | None) -> Tensor:
        """Move one decoded block from the anchor frame to the advancing refine frame it was trained in.

        The model denormalizes the whole block `(R, H * patch_size, Q)` with the frame of the
        last context patch. The first horizon patch keeps that frame; each later one is
        re-denormalized with :func:`timestral.scaler.advancing_block_frames`. Covariate rows
        keep their quantiles.
        """
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        _, loc, scale = self.model.scale(context, mask)
        loc0, scale0 = loc[:, -1], scale[:, -1]
        median = quantiles[..., self.median_index].unflatten(-1, (n_pred, ps)).to(torch.float64)
        sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
        loc_j, scale_j = advancing_block_frames(context, mask, sinh_median, self.config.minimum_scale)
        first = torch.zeros(loc_j.shape, dtype=torch.bool, device=loc_j.device)
        first[:, 0] = True
        loc_j = torch.where(first, loc0[:, None], loc_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
        scale_j = torch.where(first, scale0[:, None], scale_j.to(quantiles.dtype)).repeat_interleave(ps, dim=-1)
        sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])
        if target_rows is not None:
            refined = torch.where(target_rows[:, None, None], refined, quantiles)
        return refined


def _negate(x: Tensor | None) -> Tensor | None:
    return -x if x is not None else None


def _cat_rows(a: Tensor | None, b: Tensor | None) -> Tensor | None:
    """Concatenate covariate rows along the channel axis, either side possibly None."""
    if a is None or b is None:
        return a if b is None else b
    return torch.cat([a, b], dim=1)
