# Copyright contributors to the Tabby project.
# ruff: noqa
"""Tabby configuration.

Tabby is a PatchTST-FM backbone plus a prompt module that is prepended to the
patch-token sequence. The two are always used together, so one config carries
both halves.

Note the two different window widths, which are easy to confuse:

* ``context_length`` (8192) is the **backbone window** -- the fixed width the
  transformer always sees, laid out as ``[left-pad | history | future]``.
* ``history_length`` (8096) is how much **history** the forecasting protocol
  feeds in. ``history_length + prediction_length`` must fit ``context_length``.
  It is also the constant the segment split anchors to, so changing it changes
  the prompt's segmentation and moves the numbers.
"""

from sktime.utils.dependencies import _safe_import

PretrainedConfig = _safe_import("transformers.configuration_utils.PretrainedConfig")


class TabbyConfig(PretrainedConfig):
    model_type = "tabby"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # ---- backbone (PatchTST-FM) ----
        context_length: int = 8192,
        patch_size: int = 16,
        num_layers: int = 20,
        d_model: int = 768,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_quantiles: int = 99,
        mask_ratio: float = 0.4,
        cpm_blocks: int = 8,
        eps: float = 1e-5,
        # ---- prompt module ----
        prompt_len: int = 160,
        prompt_init_mode: str = "anchor_delta",
        prompt_gate_init: float = 5.0,
        context_aware: bool = True,
        n_stats: int = 10,
        ctx_rank: int = 4,
        ctx_hidden: int = 32,
        ctx_mlp_layers: int = 2,
        ctx_gate_init: float = 5.0,
        segmented: bool = True,
        max_segments: int = 16,
        seg_attn_dim: int = 64,
        seg_attn_heads: int = 4,
        query_from_base: bool = True,
        seg_mode: str = "adaptive_horizon",
        seg_fixed_len: int = 256,
        seg_len_k: int = 2,
        min_seg_len: int = 48,
        seg_relative_pos: bool = True,
        season_period: int = 48,
        # ---- forecasting protocol ----
        history_length: int = 8096,
        prediction_length: int = 96,
        norm_mode: str = "inhouse_asinh",
        min_forecast_span: int = 0,
        **kwargs,
    ):
        if context_length % patch_size != 0:
            raise ValueError("context_length must be divisible by patch_size.")
        if d_model % head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim.")
        if history_length + prediction_length > context_length:
            raise ValueError(
                f"history_length({history_length}) + prediction_length({prediction_length}) "
                f"must fit the backbone window {context_length}."
            )

        self.context_length = context_length
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_quantiles = num_quantiles
        self.mask_ratio = mask_ratio
        self.cpm_blocks = cpm_blocks
        self.eps = eps

        self.prompt_len = prompt_len
        self.prompt_init_mode = prompt_init_mode
        self.prompt_gate_init = prompt_gate_init
        self.context_aware = context_aware
        self.n_stats = n_stats
        self.ctx_rank = ctx_rank
        self.ctx_hidden = ctx_hidden
        self.ctx_mlp_layers = ctx_mlp_layers
        self.ctx_gate_init = ctx_gate_init
        self.segmented = segmented
        self.max_segments = max_segments
        self.seg_attn_dim = seg_attn_dim
        self.seg_attn_heads = seg_attn_heads
        self.query_from_base = query_from_base
        self.seg_mode = seg_mode
        self.seg_fixed_len = seg_fixed_len
        self.seg_len_k = seg_len_k
        self.min_seg_len = min_seg_len
        self.seg_relative_pos = seg_relative_pos
        self.season_period = season_period

        self.history_length = history_length
        self.prediction_length = prediction_length
        self.norm_mode = norm_mode
        self.min_forecast_span = min_forecast_span

        super().__init__(**kwargs)

    @property
    def num_patches(self) -> int:
        return self.context_length // self.patch_size

    @property
    def num_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def quantile_levels(self):
        """The 99 levels the head emits, in the order of ``quantile_preds``."""
        K = self.num_quantiles
        if K == 99:
            return [i / 100.0 for i in range(1, 100)]
        return [i / (K + 1) for i in range(1, K + 1)]
