# ruff: noqa
# Compatibility adapter between `train_mixup.py` and the HF-style model files.
#
# The pretraining script expects (and previously had) a module
# `models/PatchTSTFM.py` exposing:
#
#   * ``PatchTSTFMConfig`` -- a **dataclass** (the trainer calls
#     ``dataclasses.asdict(cfg)`` when checkpointing) with fields
#     ``context_length, patch_size, num_layers, d_model, head_dim, dropout,
#     num_quantiles, mask_ratio, cpm_blocks, eps`` and properties
#     ``num_patches`` / ``num_heads``.  ``utils/input_preprocessing.py`` reads
#     ``cfg.patch_size``, ``cfg.num_patches``, ``cfg.mask_ratio``,
#     ``cfg.cpm_blocks`` and ``cfg.eps`` from it.
#
#   * ``PatchTSTFM`` -- an ``nn.Module`` with
#     ``forward(x_norm, union_mask, patch_key_padding_mask) -> q_hat`` where
#     ``x_norm``  is FloatTensor[B, T], already asinh-normalized outside the
#                 model and zeroed at masked/missing/padded positions,
#     ``union_mask`` is BoolTensor[B, T] (pred | missing | padding),
#     ``patch_key_padding_mask`` is BoolTensor[B, N] (True = fully padded
#                 patch), and
#     ``q_hat``  is FloatTensor[B, T, num_quantiles], monotone in the last
#                 dim, in the *normalized* space (the pinball loss and the
#                 inverse transform live in the training script / utils).
#
# The refactored model files (``configuration_patchtst_fm.py`` /
# ``modeling_patchtst_fm.py``) expose a different vocabulary
# (``d_patch, n_layer, n_head, num_quantile, ...``) and a
# ``PatchTSTFMModel.forward`` that additionally performs RevIN normalization
# internally -- which would double-normalize the trainer's already-normalized
# input.  This adapter therefore (i) maps the trainer's config fields onto the
# HF config, and (ii) drives the *sub-modules* of ``PatchTSTFMModel``
# (patch embedding, positional embedding, transformer blocks, monotone
# quantile head) exactly as ``PatchTSTFMModel.decode`` does, while leaving
# normalization to ``utils.input_preprocessing`` as the pretraining logic
# requires.  Because the trainable module is held under ``self.backbone``,
# checkpoints saved by ``train_mixup.py`` have ``backbone.*`` keys and load
# directly into ``PatchTSTFMForPrediction`` for inference.
#
# One deliberate difference from ``PatchTSTFMModel.decode``: attention is
# masked on *keys only*.  ``decode`` masks padded queries as well, so a padded
# patch's attention row is entirely ``-inf``; depending on the PyTorch
# version/SDPA backend, fully masked rows may yield NaN outputs (some builds
# zero-fill them instead).  At inference the padded region is sliced away, but
# during pretraining the loss is computed as ``loss * loss_mask`` and
# ``NaN * 0 = NaN``, so a single left-padded real series could poison the
# batch loss on such backends.  Key-only masking (the convention of the
# original pretraining model, cf. ``src_key_padding_mask`` in
# ``nn.TransformerEncoder``) keeps padded-query outputs finite on every
# backend; they carry no gradient because ``pred_mask`` never overlaps
# padding.

from __future__ import annotations

from dataclasses import dataclass

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")

from .configuration_patchtst_fm import PatchTSTFMConfig as HFPatchTSTFMConfig
from .modeling_patchtst_fm import PatchTSTFMModel


@dataclass
class PatchTSTFMConfig:
    """Trainer-facing configuration (dataclass, as `asdict` requires)."""

    context_length: int = 8192
    patch_size: int = 16
    num_layers: int = 20
    d_model: int = 1024
    head_dim: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_quantiles: int = 99
    mask_ratio: float = 0.4
    cpm_blocks: int = 8
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.context_length % self.patch_size != 0:
            raise ValueError("context_length must be divisible by patch_size.")
        if self.d_model % self.head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim.")

    @property
    def num_patches(self) -> int:
        return self.context_length // self.patch_size

    @property
    def num_heads(self) -> int:
        return self.d_model // self.head_dim

    def to_hf_config(self) -> HFPatchTSTFMConfig:
        """Map trainer field names onto the HF-style config vocabulary."""
        return HFPatchTSTFMConfig(
            context_length=self.context_length,
            d_patch=self.patch_size,
            d_model=self.d_model,
            n_head=self.num_heads,
            n_layer=self.num_layers,
            pretrain_mask_ratio=self.mask_ratio,
            pretrain_mask_cont=self.cpm_blocks,
            num_quantile=self.num_quantiles,
            dropout=self.dropout,  # read via getattr in modeling_patchtst_fm
        )


class PatchTSTFM(nn.Module):
    """Pretraining wrapper around the HF-style ``PatchTSTFMModel`` backbone."""

    def __init__(self, cfg: PatchTSTFMConfig):
        super().__init__()
        self.config = cfg
        self.backbone = PatchTSTFMModel(cfg.to_hf_config())

    def forward(
        self,
        x_norm: torch.Tensor,
        union_mask: torch.Tensor,
        patch_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized quantile predictions of shape [B, T, K].

        Mirrors ``PatchTSTFMModel.decode`` (same sub-modules, same monotone
        quantile construction), with key-only attention padding masking; see
        the module docstring for the rationale.
        """
        cfg = self.config
        B, T = x_norm.shape
        if T != cfg.context_length:
            raise ValueError(f"Expected T={cfg.context_length}, got {T}.")

        L, N, K = cfg.patch_size, cfg.num_patches, cfg.num_quantiles
        bb = self.backbone

        value_patch = x_norm.reshape(B, N, L)
        mask_patch = union_mask.reshape(B, N, L).to(x_norm.dtype)

        # Same input convention as PatchTSTFMModel.decode: cat([values, 1 - mask]).
        h = bb.in_layer(torch.cat([value_patch, 1.0 - mask_patch], dim=-1))
        h = bb.pos_embed(h)

        # Key-only boolean attention mask, broadcast to (B, n_head, N, N):
        # True = position may be attended to.
        attn_mask = ~patch_key_padding_mask.bool()[:, None, None, :]
        for block in bb.blocks:
            h = block(h, attn_mask)

        h = bb.out_layer(h)

        # Monotone quantile head, identical to PatchTSTFMModel.decode.
        q_raw = h.reshape(B, N, K + 1, L).permute(0, 2, 1, 3)  # (B, K+1, N, L)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            nn.functional.softplus(q_raw[:, 1:, :, :]) / K, dim=1
        )  # (B, K, N, L)

        q_hat = q.permute(0, 2, 3, 1).reshape(B, T, K)  # (B, T, K)
        return q_hat
