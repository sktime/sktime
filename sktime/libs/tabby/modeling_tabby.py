# Copyright contributors to the Tabby project.
# ruff: noqa
"""Tabby: a prompted PatchTST-FM time-series foundation model.

Self-contained inference model. The forward pass is the inference half of
``tabby_prompt.model.PromptedPatchTSTFM.forward`` from the training repository,
copied rather than reimplemented so the released weights reproduce the published
numbers exactly. Two branches of the original are dropped because they cannot be
reached at inference with these weights:

* the pinball-loss branch, which only runs when ``future_target`` is given;
* the long-history prompt branch, which requires ``long_ctx_len > history_length``
  and is disabled in this checkpoint (``long_ctx_len = 0``).

Everything else -- the ``[left-pad | history | future]`` window layout, the
visible-history-only asinh normalization, key-only attention masking, and the
monotone cumulative-softplus quantile head -- is unchanged.
"""

from typing import List, Optional, Union

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")
PreTrainedModel = _safe_import("transformers.modeling_utils.PreTrainedModel")

from .configuration_tabby import TabbyConfig
from .input_preprocessing import mask_aware_normalize_for_inference
from .patchtstfm_adapter import PatchTSTFM, PatchTSTFMConfig
from .prompt_blocks import (
    Chronos2ContextPromptGenerator,
    Chronos2StatsExtractor,
    SegmentedStatsExtractor,
    SegmentRefiner,
)


class TabbyPreTrainedModel(PreTrainedModel):
    config_class = TabbyConfig
    base_model_prefix = "tabby"
    main_input_name = "context"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        # Tabby is only ever instantiated from released weights; this exists so
        # `PreTrainedModel.post_init` has something to call.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


class TabbyForForecasting(TabbyPreTrainedModel):
    """Probabilistic univariate forecaster.

    >>> model = TabbyForForecasting.from_pretrained("<ns>/Tabby").eval()
    >>> q = model.predict(history, prediction_length=96)   # [B, 99, 96]
    """

    def __init__(self, config: TabbyConfig):
        super().__init__(config)

        model_cfg = PatchTSTFMConfig(
            context_length=config.context_length,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            d_model=config.d_model,
            head_dim=config.head_dim,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_quantiles=config.num_quantiles,
            mask_ratio=config.mask_ratio,
            cpm_blocks=config.cpm_blocks,
            eps=config.eps,
        )
        self.model_cfg = model_cfg
        self.backbone = PatchTSTFM(model_cfg)

        d = config.d_model
        M = config.prompt_len

        self.register_buffer(
            "quantile_levels_t",
            torch.tensor(config.quantile_levels, dtype=torch.float32),
        )

        if config.prompt_init_mode == "anchor_delta":
            self.register_buffer("prompt_anchor", torch.zeros(M, d))
            self.prompt_delta = nn.Parameter(torch.zeros(M, d))
            self.prompt_gate = nn.Parameter(
                torch.tensor(float(config.prompt_gate_init))
            )
        elif config.prompt_init_mode == "direct":
            self.prompt_embedding = nn.Parameter(torch.zeros(M, d))
        else:
            raise ValueError(f"Unknown prompt_init_mode: {config.prompt_init_mode}")

        if config.context_aware:
            self.ctx_gate = nn.Parameter(torch.tensor(float(config.ctx_gate_init)))
            self.stats_extractor = Chronos2StatsExtractor(n_stats=config.n_stats)
            self.ctx_generator = Chronos2ContextPromptGenerator(
                feat_dim=config.n_stats,
                prompt_len=M,
                d_model=d,
                rank=config.ctx_rank,
                hidden_dim=config.ctx_hidden,
                n_layers=config.ctx_mlp_layers,
            )
            if config.segmented:
                self.seg_extractor = SegmentedStatsExtractor(
                    n_stats=config.n_stats,
                    max_segments=config.max_segments,
                    seg_mode=config.seg_mode,
                    seg_fixed_len=config.seg_fixed_len,
                    seg_len_k=config.seg_len_k,
                    min_seg_len=config.min_seg_len,
                    prediction_length=config.prediction_length,
                    season_period=config.season_period,
                    context_length=config.history_length,
                )
                self.seg_refiner = SegmentRefiner(
                    n_stats=config.n_stats,
                    d_model=d,
                    prompt_len=M,
                    n_segments=config.max_segments,
                    d_attn=config.seg_attn_dim,
                    n_heads=config.seg_attn_heads,
                    query_from_base=config.query_from_base,
                    relative_pos=config.seg_relative_pos,
                )

        self.post_init()

    # ------------------------------------------------------------------ props
    @property
    def num_quantiles(self) -> int:
        return self.model_cfg.num_quantiles

    @property
    def quantile_levels(self) -> List[float]:
        return [float(q) for q in self.quantile_levels_t.tolist()]

    def get_prompt_embedding(self) -> torch.Tensor:
        if self.config.prompt_init_mode == "anchor_delta":
            return (
                self.prompt_anchor + torch.sigmoid(self.prompt_gate) * self.prompt_delta
            )
        return self.prompt_embedding

    # ------------------------------------------------------------ window build
    def _build_window(self, context, context_mask, past_is_pad, H):
        """``[left-pad | history | future]`` over the fixed backbone width Tm."""
        device = context.device
        B, Tc = context.shape
        Tm = self.model_cfg.context_length
        Hs = max(H, int(self.config.min_forecast_span or 0))

        cap = Tm - Hs
        if Tc > cap:  # keep the most recent history
            context = context[:, -cap:]
            context_mask = context_mask[:, -cap:]
            if past_is_pad is not None:
                past_is_pad = past_is_pad[:, -cap:]
            Tc = cap
        left = Tm - Tc - Hs

        x = torch.zeros(B, Tm, device=device, dtype=torch.float32)
        observed = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        padding = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        pred = torch.zeros(B, Tm, dtype=torch.bool, device=device)

        padding[:, :left] = True
        hist_sl = slice(left, left + Tc)
        fut_sl = slice(Tm - Hs, Tm)
        tgt_sl = slice(Tm - Hs, Tm - Hs + H)

        ctx_finite = torch.isfinite(context)
        obs_hist = ctx_finite & (context_mask > 0)
        if past_is_pad is not None:
            pad_hist = past_is_pad.bool()
            obs_hist = obs_hist & ~pad_hist
            padding[:, hist_sl] = pad_hist
        x[:, hist_sl] = torch.nan_to_num(context.float(), nan=0.0)
        observed[:, hist_sl] = obs_hist
        pred[:, fut_sl] = True

        return x, observed, padding, pred, hist_sl, tgt_sl

    # ------------------------------------------------------------ normalization
    def _normalize(self, x, observed, pred, padding):
        x_norm, mean, std, union_mask, patch_pad = mask_aware_normalize_for_inference(
            x=x,
            observed_mask=observed,
            pred_mask=pred,
            padding_mask=padding,
            cfg=self.model_cfg,
        )
        if self.config.norm_mode == "inhouse_asinh":
            return x_norm, mean, std, union_mask, patch_pad
        if self.config.norm_mode != "revin_official":
            raise ValueError(f"unknown norm_mode {self.config.norm_mode}")
        # RevIN._get_statistics(x, mask=union_mask) verbatim (fp32, no eps)
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            unmask = (~union_mask).float()
            count = unmask.sum(dim=1, keepdim=True).clamp(min=1)
            r_mean = (xf * unmask).sum(dim=1, keepdim=True) / count
            r_std = (
                (((xf - r_mean) * unmask) ** 2).sum(dim=1, keepdim=True) / count
            ).sqrt()
            r_std = torch.where(r_std > 1e-5, r_std, torch.ones_like(r_std))
            xn = torch.asinh((xf - r_mean) / r_std)
            xn = torch.where(union_mask, torch.zeros_like(xn), xn)
        return xn, r_mean, r_std, union_mask, patch_pad

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        past_is_pad: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,
        return_dict: bool = True,
    ):
        """
        context:            [B, Tc] raw history, NaN allowed at unobserved steps
        context_mask:       [B, Tc] 1 = valid observed history (default: isfinite)
        past_is_pad:        [B, Tc] bool, left-padding from a batch loader
        prediction_length:  horizon; defaults to ``config.prediction_length``

        Returns ``{"quantile_preds": [B, K, H]}`` in the original data scale,
        with ``K = 99`` levels ordered as ``config.quantile_levels``.
        """
        cfg = self.config
        B = context.shape[0]
        M = cfg.prompt_len
        H = (
            int(prediction_length)
            if prediction_length is not None
            else cfg.prediction_length
        )
        Tm = self.model_cfg.context_length
        L = self.model_cfg.patch_size
        N = self.model_cfg.num_patches
        K = self.model_cfg.num_quantiles
        Hs = max(H, int(cfg.min_forecast_span or 0))
        if not (1 <= Hs <= Tm - L):
            raise ValueError(f"forecast span={Hs} (H={H}) out of range for window {Tm}")

        if context_mask is None:
            context_mask = torch.isfinite(context).float()

        x, observed, padding, pred, hist_sl, tgt_sl = self._build_window(
            context, context_mask, past_is_pad, H
        )
        x_norm, mean, std, union_mask, patch_pad = self._normalize(
            x, observed, pred, padding
        )

        # ---- prompt pathway (on the right-aligned history slice) ----
        prompt = self.get_prompt_embedding().unsqueeze(0).expand(B, M, -1)
        if cfg.context_aware and M > 0:
            hist_vals = x_norm[:, hist_sl]  # zeros at invisible
            hist_mask = (~union_mask[:, hist_sl]).float()  # 1 = visible history
            global_stats = self.stats_extractor(hist_vals, hist_mask)  # [B, n_stats]
            base_prompt = self.ctx_generator(global_stats)  # [B, M, d]
            if cfg.segmented:
                seg_matrix, seg_valid = self.seg_extractor(hist_vals, hist_mask)
                ctx_prompt = base_prompt + self.seg_refiner(
                    seg_matrix, seg_valid, base_prompt
                )
            else:
                ctx_prompt = base_prompt
            prompt = prompt + torch.sigmoid(self.ctx_gate) * ctx_prompt

        # ---- backbone drive: patch embed + M prompt tokens ----
        bb = self.backbone.backbone
        value_patch = x_norm.reshape(B, N, L)
        mask_patch = union_mask.reshape(B, N, L).to(x_norm.dtype)

        h = bb.in_layer(torch.cat([value_patch, 1.0 - mask_patch], dim=-1))
        h = bb.pos_embed(h)  # prompts get NO positional embedding
        if M > 0:
            h = torch.cat([prompt.to(h.dtype), h], dim=1)

        key_ok = ~patch_pad.bool()  # [B, N]; True = attendable (key-only)
        if M > 0:
            prompt_ok = torch.ones(B, M, dtype=torch.bool, device=h.device)
            key_ok = torch.cat([prompt_ok, key_ok], dim=1)
        attn_mask = key_ok[:, None, None, :]
        for block in bb.blocks:
            h = block(h, attn_mask)
        if M > 0:
            h = h[:, M:]
        h = bb.out_layer(h)

        # monotone quantile head: base level + cumulative softplus increments
        q_raw = h.reshape(B, N, K + 1, L).permute(0, 2, 1, 3)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            nn.functional.softplus(q_raw[:, 1:, :, :]) / K, dim=1
        )
        q_hat = q.permute(0, 2, 3, 1).reshape(B, Tm, K)

        q_f = q_hat[:, tgt_sl, :]  # [B, H, K] normalized
        q_out = torch.sinh(q_f.float()) * std.unsqueeze(-1) + mean.unsqueeze(-1)
        q_out = q_out.permute(0, 2, 1)  # [B, K, H]

        if not return_dict:
            return (q_out,)
        return {"quantile_preds": q_out}

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        context: Union[torch.Tensor, List, "object"],
        prediction_length: Optional[int] = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Forecast one or more univariate series.

        ``context`` may be a 1-D sequence, a 2-D ``[B, Tc]`` array/tensor, or a
        list of equal-length sequences. Shorter series should be left-padded with
        NaN, which is treated as unobserved.

        Returns ``[B, 99, H]`` on the model's device, in the input's own scale.
        Use ``model.quantile_levels`` for the level of each row; index 49 is the
        median.
        """
        self.eval()
        if not torch.is_tensor(context):
            context = torch.as_tensor(context, dtype=torch.float32)
        context = context.to(dtype=torch.float32)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.dim() != 2:
            raise ValueError(
                f"context must be 1-D or 2-D [B, Tc], got {tuple(context.shape)}"
            )

        device = next(self.parameters()).device
        outs = []
        for i in range(0, context.shape[0], batch_size):
            chunk = context[i : i + batch_size].to(device)
            outs.append(
                self(context=chunk, prediction_length=prediction_length)[
                    "quantile_preds"
                ]
            )
        return torch.cat(outs, dim=0)
