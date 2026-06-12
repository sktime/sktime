# Copyright contributors to the TSFM project
#
# This code is based on layers and components from the PatchTSMixer model in the HuggingFace Transformers
# Library: https://github.com/huggingface/transformers/blob/main/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py
"""PyTorch TinyTimeMixer model."""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.time_series_utils import (
    NegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_tinytimemixer import TinyTimeMixerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TinyTimeMixerConfig"


TINYTIMEMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = []


TINYTIMEMIXER_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TinyTimeMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TINYTIMEMIXER_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def update_patch_mask(patch_mask: torch.Tensor, K: int, mode: str = "prepend") -> torch.Tensor:
    """
    Add K patches (all valid = all False) either at the beginning or end
    of a patch-level mask.

    Args:
        patch_mask (Tensor): Boolean mask of shape (B, C, N),
            where B = batch size, C = channels, N = number of patches.
            True means this patch is masked (invalid),
            False means valid.
        K (int): Number of patches (all valid = all False) to add.
        mode (str): "prepend" to add at the beginning,
                    "postpend" to add at the end.

    Returns:
        Tensor: Updated boolean mask of shape (B, C, N + K).
    """
    if patch_mask is None:
        return patch_mask

    B, C, N = patch_mask.shape
    extra_mask = torch.zeros((B, C, K), dtype=torch.bool, device=patch_mask.device)

    if mode == "prepend":
        new_mask = torch.cat([extra_mask, patch_mask], dim=2)
    elif mode == "postpend":
        new_mask = torch.cat([patch_mask, extra_mask], dim=2)
    else:
        raise ValueError("mode must be 'prepend' or 'postpend'")

    return new_mask


class MultiPinballLoss(nn.Module):
    """
    Quantile (pinball) loss for predictions shaped as [B, Q, T, C]
    against targets shaped as [B, T, C].

    Taus:
      - if config.quantile_levels is None -> torch.linspace(0.1,0.9,Q) (preserves old default)
      - else -> sorted(config.quantile_levels)

    Optional horizon weighting via `horizon_weights` (shape [T]).

    Optional width regularization (training-time sharpness control):
      - config.penalize_large_width_ratio: float (0 disables)
      - config.width_penalty_mode: "boundary" or "wis" (default: "boundary")

        "boundary": penalize only outer width (q_max - q_min)
        "wis": penalize mean of symmetric interval widths across all pairs
               (q_high(i) - q_low(i)) for i=0..k-1 where Q=2k+1
    """

    def __init__(self, config, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

        qlist = getattr(config, "quantile_levels", None)
        # self._use_default_linspace = qlist is None
        # to allow backward compatibility
        self._use_default_linspace = qlist is None or qlist == [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]
        self.static_taus = None if self._use_default_linspace else [float(q) for q in sorted(qlist)]

        self.penalize_large_width_ratio = float(getattr(config, "penalize_large_width_ratio", 0.0))
        self.width_penalty_mode = str(getattr(config, "width_penalty_mode", "boundary")).lower()
        if self.width_penalty_mode not in ("boundary", "wis"):
            raise ValueError(f"width_penalty_mode must be 'boundary' or 'wis', got: {self.width_penalty_mode}")

    @staticmethod
    def build_two_stage_horizon_weights(
        T: int,
        cut_ratio: float = 0.30,
        floor: float = 0.30,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        cut = int(round(cut_ratio * T))
        cut = max(1, min(T, cut))

        w = torch.ones(T, device=device, dtype=dtype)
        if cut < T:
            w[cut:] = torch.linspace(1.0, float(floor), T - cut, device=device, dtype=dtype)

        return w / (w.mean() + 1e-12)

    def forward(
        self,
        pred: torch.Tensor,  # [B, Q, T, C]
        target: torch.Tensor,  # [B, T, C]
        taus=None,
        horizon_weights: torch.Tensor = None,  # [T] optional
    ) -> torch.Tensor:
        # ----------------------------
        # Resolve taus
        # ----------------------------
        if taus is None:
            if self._use_default_linspace:
                taus = torch.linspace(0.1, 0.9, pred.size(1), device=pred.device, dtype=pred.dtype)
            else:
                taus = torch.as_tensor(self.static_taus, device=pred.device, dtype=pred.dtype)
        else:
            taus = torch.as_tensor(taus, device=pred.device, dtype=pred.dtype)

        if taus.numel() != pred.size(1):
            raise ValueError(
                f"taus length ({taus.numel()}) must match pred.size(1) ({pred.size(1)}). "
                f"config.quantile_levels={self.static_taus}"
            )

        # ----------------------------
        # Pinball loss per element
        # ----------------------------
        target_exp = target.unsqueeze(1)  # [B,1,T,C]
        taus_exp = taus.view(1, -1, 1, 1)  # [1,Q,1,1]
        # print("target:", target_exp.shape, pred.shape)
        e = target_exp - pred  # [B,Q,T,C]
        per_elem = torch.maximum(taus_exp * e, (taus_exp - 1.0) * e)  # [B,Q,T,C]

        # Quantile weights (keep your behavior)
        per_elem = per_elem * torch.ones_like(taus).view(1, -1, 1, 1)

        # Horizon weights
        hw = None
        if horizon_weights is not None:
            hw = horizon_weights.to(device=pred.device, dtype=pred.dtype)
            if hw.ndim != 1 or hw.numel() != pred.size(2):
                raise ValueError("horizon_weights must be shape [T] and match pred.size(2)")
            per_elem = per_elem * hw.view(1, 1, -1, 1)

        # Reduce pinball
        if self.reduction == "none":
            pinball = per_elem
        elif self.reduction == "sum":
            pinball = per_elem.sum()
        elif self.reduction == "mean":
            pinball = per_elem.mean()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        # ----------------------------
        # Width regularization (optional)
        # ----------------------------
        if self.penalize_large_width_ratio > 0.0:
            Q = pred.size(1)
            if Q < 3:
                raise ValueError("Width penalty requires at least 3 quantiles (Q>=3).")

            if self.width_penalty_mode == "boundary":
                # outermost interval only: q_max - q_min
                width = pred[:, -1, :, :] - pred[:, 0, :, :]  # [B,T,C]
                if hw is not None:
                    width = width * hw.view(1, -1, 1)
                width_pen = width.sum() if self.reduction == "sum" else width.mean()

            else:  # "wis"
                # average over all symmetric interval widths:
                # (q_{high} - q_{low}) for pairs (0,-1), (1,-2), ...
                if Q % 2 != 1:
                    raise ValueError(f"WIS-like width penalty expects odd Q (median present). Got Q={Q}.")
                k = (Q - 1) // 2

                # accumulate widths for each pair
                widths = []
                for i in range(k):
                    w_i = pred[:, -(i + 1), :, :] - pred[:, i, :, :]  # [B,T,C]
                    if hw is not None:
                        w_i = w_i * hw.view(1, -1, 1)
                    widths.append(w_i.mean() if self.reduction != "sum" else w_i.sum())

                width_pen = torch.stack(widths).mean()

            pinball = pinball + self.penalize_large_width_ratio * width_pen

        return pinball


# class MultiPinballLoss(nn.Module):
#     """
#     Quantile (pinball) loss for predictions shaped as [B, Q, T, C]
#     against targets shaped as [B, T, C].

#     Taus are taken from config.quantile_levels by default (sorted).
#     Falls back to [0.1..0.9] if not present.

#     Adds optional horizon weighting via `horizon_weights` (shape [T]).
#     """

#     def __init__(self, config, reduction: str = "mean"):
#         super().__init__()
#         self.reduction = reduction

#         qlist = getattr(config, "quantile_levels", None)

#         if qlist is None:
#             self.static_taus = None
#         else:
#             self.static_taus = [float(q) for q in sorted(qlist)]

#         # if qlist is None:
#         #     qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#         # # keep deterministic order
#         # self.static_taus = [float(q) for q in sorted(qlist)]

#     @staticmethod
#     def build_two_stage_horizon_weights(
#         T: int,
#         cut_ratio: float = 0.30,  # auto "30%" by default
#         floor: float = 0.30,  # long-horizon still has weight >= floor
#         device=None,
#         dtype=None,
#     ) -> torch.Tensor:
#         """
#         Two-stage schedule:
#           - t < cut: weight = 1.0
#           - t >= cut: linearly decays from 1.0 to `floor`

#         Returns weights normalized to mean=1 for stable loss scale.
#         """
#         cut = int(round(cut_ratio * T))
#         cut = max(1, min(T, cut))

#         w = torch.ones(T, device=device, dtype=dtype)
#         if cut < T:
#             w[cut:] = torch.linspace(
#                 1.0, float(floor), T - cut, device=device, dtype=dtype
#             )

#         # normalize to keep loss magnitude comparable
#         w = w / (w.mean() + 1e-12)
#         return w

#     def forward(
#         self,
#         pred: torch.Tensor,  # [B, Q, T, C]
#         target: torch.Tensor,  # [B, T, C]
#         taus=None,
#         horizon_weights: torch.Tensor = None,  # [T] optional
#     ) -> torch.Tensor:

#         # Resolve taus
#         if self.static_taus is None:
#             taus = torch.linspace(
#                 0.1, 0.9, pred.size(1), device=pred.device, dtype=pred.dtype
#             )
#         else:
#             taus = torch.as_tensor(
#                 self.static_taus, device=pred.device, dtype=pred.dtype
#             )

#         # Safety: ensure Q matches pred.size(1) when using default taus
#         if taus.numel() != pred.size(1):
#             raise ValueError(
#                 f"taus length ({taus.numel()}) must match pred.size(1) ({pred.size(1)}). "
#                 f"config.quantile_levels={self.static_taus}"
#             )

#         # Broadcast shapes
#         target_exp = target.unsqueeze(1)  # [B, 1, T, C]
#         taus_exp = taus.view(1, -1, 1, 1)  # [1, Q, 1, 1]

#         # Pinball loss per element
#         e = target_exp - pred  # [B, Q, T, C]
#         per_elem = torch.maximum(taus_exp * e, (taus_exp - 1.0) * e)  # [B,Q,T,C]

#         # Quantile weights (keep your current behavior)
#         wq = torch.ones_like(taus)  # [Q]
#         per_elem = per_elem * wq.view(1, -1, 1, 1)

#         # Horizon weights (optional)
#         if horizon_weights is not None:
#             hw = horizon_weights.to(device=pred.device, dtype=pred.dtype)
#             if hw.ndim != 1 or hw.numel() != pred.size(2):
#                 raise ValueError(
#                     "horizon_weights must be shape [T] and match pred.size(2)"
#                 )
#             per_elem = per_elem * hw.view(1, 1, -1, 1)

#         # Reduction
#         if self.reduction == "none":
#             return per_elem
#         if self.reduction == "sum":
#             return per_elem.sum()
#         if self.reduction == "mean":
#             return per_elem.mean()

#         raise ValueError(f"Unsupported reduction: {self.reduction}")


# class MultiPinballLoss(nn.Module):
#     """
#     Quantile (pinball) loss for predictions shaped as [B, Q, T, C]
#     against targets shaped as [B, T, C].

#     Adds optional horizon weighting via `horizon_weights` (shape [T]).
#     """

#     def __init__(self, config, taus=None, reduction: str = "mean"):
#         super().__init__()
#         self.reduction = reduction
#         self.static_taus = None if taus is None else list(taus)

#     @staticmethod
#     def build_two_stage_horizon_weights(
#         T: int,
#         cut_ratio: float = 0.30,  # auto "30%" by default
#         floor: float = 0.30,  # long-horizon still has weight >= floor
#         device=None,
#         dtype=None,
#     ) -> torch.Tensor:
#         """
#         Two-stage schedule:
#           - t < cut: weight = 1.0
#           - t >= cut: linearly decays from 1.0 to `floor`

#         Returns weights normalized to mean=1 for stable loss scale.
#         """
#         cut = int(round(cut_ratio * T))
#         cut = max(1, min(T, cut))

#         w = torch.ones(T, device=device, dtype=dtype)
#         if cut < T:
#             w[cut:] = torch.linspace(
#                 1.0, float(floor), T - cut, device=device, dtype=dtype
#             )
#         # normalize to keep loss magnitude comparable
#         w = w / (w.mean() + 1e-12)
#         return w

#     def forward(
#         self,
#         pred: torch.Tensor,  # [B, Q, T, C]
#         target: torch.Tensor,  # [B, T, C]
#         taus=None,
#         horizon_weights: torch.Tensor = None,  # [T] optional
#     ) -> torch.Tensor:

#         # Resolve taus
#         if taus is None:
#             if self.static_taus is not None:
#                 taus = torch.as_tensor(
#                     self.static_taus, device=pred.device, dtype=pred.dtype
#                 )
#             else:
#                 taus = torch.linspace(
#                     0.1, 0.9, pred.size(1), device=pred.device, dtype=pred.dtype
#                 )
#         else:
#             taus = torch.as_tensor(taus, device=pred.device, dtype=pred.dtype)

#         # Broadcast shapes
#         target_exp = target.unsqueeze(1)  # [B, 1, T, C]
#         taus_exp = taus.view(1, -1, 1, 1)  # [1, Q, 1, 1]

#         # Pinball loss per element
#         e = target_exp - pred  # [B, Q, T, C]
#         per_elem = torch.maximum(taus_exp * e, (taus_exp - 1.0) * e)  # [B,Q,T,C]

#         # Quantile weights (keep your current behavior)
#         wq = torch.ones_like(taus)  # [Q]
#         wq_exp = wq.view(1, -1, 1, 1)  # [1,Q,1,1]
#         per_elem = per_elem * wq_exp

#         # Horizon weights (new)
#         if horizon_weights is not None:
#             hw = horizon_weights.to(device=pred.device, dtype=pred.dtype)
#             assert hw.ndim == 1 and hw.numel() == pred.size(
#                 2
#             ), "horizon_weights must be shape [T]"
#             hw_exp = hw.view(1, 1, -1, 1)  # [1,1,T,1]
#             per_elem = per_elem * hw_exp

#         # Reduction
#         if self.reduction == "none":
#             return per_elem
#         elif self.reduction == "sum":
#             return per_elem.sum()
#         elif self.reduction == "mean":
#             return per_elem.mean()
#         else:
#             raise ValueError(f"Unsupported reduction: {self.reduction}")


def weighted_l1_over_horizon(y_hat: torch.Tensor, target: torch.Tensor, horizon_weights: torch.Tensor = None):
    """
    y_hat, target: [B, T, C]
    horizon_weights: [T] optional
    """
    l1 = (y_hat - target).abs()  # [B,T,C]
    if horizon_weights is not None:
        hw = horizon_weights.to(device=y_hat.device, dtype=y_hat.dtype)
        assert hw.ndim == 1 and hw.numel() == y_hat.size(1), "horizon_weights must be shape [T]"
        l1 = l1 * hw.view(1, -1, 1)
    return l1.mean()


class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        """
        Initialize the Pinball Loss for multidimensional tensors.

        Args:
        quantile (float): The desired quantile (e.g., 0.5 for median, 0.9 for 90th percentile).
        """
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, predictions, targets):
        """
        Compute the Pinball Loss for shape [b, seq_len, channels].

        Args:
        predictions (torch.Tensor): Predicted values, shape [b, seq_len, channels].
        targets (torch.Tensor): Ground truth values, shape [b, seq_len, channels].

        Returns:
        torch.Tensor: The mean pinball loss over all dimensions.
        """
        errors = targets - predictions

        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)

        return loss.mean()


class TinyTimeMixerGatedAttention(nn.Module):
    """
    Backward-compatible gated attention operating on last dim: x[..., C].

    Config:
      - config.gate_mode   : {"softmax", "sigmoid", "glu", "group_sigmoid"}
                             default = "softmax"
      - config.gate_groups : int (default = 8, used only for group_sigmoid)
      - config.use_register_context_gating : bool (default = False)
            If True, the gate will depend on both x and a context tensor
            (typically derived from register tokens): [x || context].
    """

    def __init__(self, config, in_size: int, out_size: int):
        super().__init__()
        assert in_size == out_size, "Gated attention expects in_size == out_size"

        self.C = in_size
        self.mode = getattr(config, "gate_mode", "softmax")  # backward compatible
        self.groups = getattr(config, "gate_groups", 8)

        # NEW: whether to use register context as extra input to the gate
        self.use_reg_context = getattr(config, "use_register_context_gating", False)

        requested = getattr(config, "gate_groups", 8)
        self.groups = self.pick_gate_groups(self.C, requested)
        self.group_size = self.C // self.groups

        if self.mode == "group_sigmoid":
            assert self.C % self.groups == 0, "channels must be divisible by gate_groups"
            self.group_size = self.C // self.groups

        # ---- Projection ----
        # Input to the gate:
        #   - normally: x      → dim = C
        #   - with context: [x || ctx] → dim = 2C
        gate_in_dim = self.C * (2 if self.use_reg_context else 1)

        if self.mode == "glu":
            # GLU needs 2*C outputs (content + gate)
            self.attn_layer = nn.Linear(gate_in_dim, 2 * self.C)
        else:
            self.attn_layer = nn.Linear(gate_in_dim, self.C)

        # Optional: call this from outside after construction if you want identity-ish init
        # self._init_identity_weights()

    def pick_gate_groups(self, C: int, requested_G: int) -> int:
        if requested_G <= 1:
            return 1

        if C % requested_G == 0:
            return requested_G

        for g in range(min(requested_G, C), 0, -1):
            if C % g == 0:
                return g

        return 1

    def _init_identity_weights(self):
        if self.mode == "softmax":
            nn.init.zeros_(self.attn_layer.weight)
            nn.init.zeros_(self.attn_layer.bias)

        elif self.mode in ("sigmoid", "group_sigmoid"):
            nn.init.zeros_(self.attn_layer.weight)
            nn.init.zeros_(self.attn_layer.bias)

        elif self.mode == "glu":
            weight = self.attn_layer.weight
            bias = self.attn_layer.bias

            a_w, b_w = weight.chunk(2, dim=0)
            a_b, b_b = bias.chunk(2, dim=0)

            nn.init.xavier_uniform_(a_w)
            nn.init.zeros_(b_w)

            nn.init.zeros_(a_b)
            nn.init.zeros_(b_b)

        else:
            raise ValueError(f"Unknown gate_mode: {self.mode}")

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        """
        x:       [..., C]
        context: [..., C] or broadcastable to x (e.g. [B, 1, C])
                 Only used if self.use_reg_context == True.

        If use_reg_context is False, 'context' is ignored and behavior is
        identical to the original implementation.
        """
        if self.use_reg_context:
            if context is None:
                raise ValueError(
                    "TinyTimeMixerGatedAttention: use_reg_context=True "
                    "but no context tensor was provided to forward()."
                )

            # Broadcast context to match x shape (except last dim)
            # e.g., context [B, 1, C] -> [B, T, C]
            # or [B, C] -> [B, T, C] depending on your call site.
            # This relies on standard PyTorch broadcasting rules.
            ctx = context
            # Ensure last dim matches
            assert ctx.shape[-1] == self.C, f"context last dim must be {self.C}, got {ctx.shape[-1]}"

            # Let broadcasting handle missing dims as long as they are compatible
            # (e.g., x [B,T,C], ctx [B,1,C] or [1,1,C])
            z = torch.cat([x, ctx.expand_as(x)], dim=-1)
        else:
            z = x  # no context → behave exactly as before

        if self.mode == "glu":
            a, b = self.attn_layer(z).chunk(2, dim=-1)
            gate = torch.sigmoid(b)
            return a * gate

        logits = self.attn_layer(z)

        if self.mode == "softmax":
            gate = F.softmax(logits, dim=-1)

        elif self.mode == "sigmoid":
            gate = torch.sigmoid(logits)

        elif self.mode == "group_sigmoid":
            g = logits.view(*z.shape[:-1], self.groups, self.group_size)
            g = g.mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(g)
            gate = gate.expand(*z.shape[:-1], self.groups, self.group_size)
            gate = gate.reshape(*z.shape[:-1], self.C)

        else:
            raise ValueError(f"Unknown gate_mode: {self.mode}")

        return x * gate


class TinyTimeMixerGatedAttentionOLD(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TinyTimeMixerCategoricalEmbeddingLayer(nn.Module):
    """ """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.categorical_vocab_size_list = config.categorical_vocab_size_list
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(vocab, config.d_model) for vocab in self.categorical_vocab_size_list]
        )
        self.number_of_categorical_variables = len(self.categorical_vocab_size_list)
        self.num_patches = config.num_patches

    def forward(self, static_categorical_values: torch.Tensor):
        """
        Parameters:
            static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`
        Returns:
            `torch.Tensor` of shape `(batch_size, number_of_categorical_variables, num_patches, d_model)`
        """
        # static_categorical_values [bs x number_of_categorical_variables]
        embedded_tensors = []

        for i in range(self.number_of_categorical_variables):
            embedded_tensor = self.embedding_layers[i](static_categorical_values[:, i].long())
            embedded_tensors.append(embedded_tensor)

        output_tensor = torch.stack(embedded_tensors, dim=1)  # bs x number_of_categorical_variables x d_model

        output_tensor = output_tensor.unsqueeze(2).repeat(
            1, 1, self.num_patches, 1
        )  # bs x number_of_categorical_variables x num_patches x d_model

        return output_tensor


class TinyTimeMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class TinyTimeMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        # positional encoding: [num_patches x d_model]
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: TinyTimeMixerConfig) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class TinyTimeMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TinyTimeMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]

            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class TinyTimeMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TinyTimeMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                config=config,
                in_size=config.num_input_channels,
                out_size=config.num_input_channels,
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


class TinyTimeMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TinyTimeMixerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                config=config, in_size=config.num_patches, out_size=config.num_patches
            )

        if config.self_attn:
            self.self_attn_layer = TinyTimeMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TinyTimeMixerNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.norm = TinyTimeMixerNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(
                config=config, in_size=config.d_model, out_size=config.d_model
            )

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class ForecastChannelHeadMixer(nn.Module):
    """ForecastChannelMixer Module to reconcile forecasts across channels with exogenous support.

    When channel_context_length is positive this mode creates a patch for every multi-variate forecast point with its surronding context
    it then flattens it and applies MLP to it.
    By this way, every forecast point learn from its pre and post surrounding context in a channel mixed way.
    Residual is added to ensure noise reduction with initial forecasts.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.fcm_context_length = config.fcm_context_length
        self.scl = 2 * self.fcm_context_length + 1

        if config.prediction_channel_indices is not None:
            self.prediction_channel_count = len(config.prediction_channel_indices)
        else:
            self.prediction_channel_count = config.num_input_channels

        if config.exogenous_channel_indices is not None:
            self.exogenous_channel_count = len(config.exogenous_channel_indices)
        else:
            self.exogenous_channel_count = 0

        self.total_channel_count = self.prediction_channel_count + self.exogenous_channel_count

        self.fcm_use_mixer = config.fcm_use_mixer

        self.exogenous_channel_indices = config.exogenous_channel_indices
        self.prediction_channel_indices = config.prediction_channel_indices
        scl_features = self.scl

        if self.fcm_use_mixer:
            # model mixer considering channel dim as patch dim for lag computation
            temp_config = copy.deepcopy(config)
            temp_config.num_patches = self.total_channel_count
            temp_config.patch_length = self.scl
            temp_config.num_input_channels = config.prediction_length
            temp_config.d_model = self.scl * 2
            temp_config.patch_stride = 1
            temp_config.num_layers = config.fcm_mix_layers
            temp_config.dropout = config.head_dropout
            temp_config.mode = "common_channel"
            temp_config.gated_attn = config.fcm_gated_attn
            temp_config.adaptive_patching_levels = 0
            self.exog_mixer = TinyTimeMixerBlock(temp_config)
            scl_features = self.scl * 2
            self.fcm_embedding = nn.Linear(temp_config.patch_length, temp_config.d_model)

        self.mlp = nn.Linear(
            self.total_channel_count * (scl_features),
            self.prediction_channel_count,
        )
        if config.fcm_gated_attn:
            self.fcm_gating_block = TinyTimeMixerGatedAttention(
                config=config,
                in_size=self.total_channel_count * (scl_features),
                out_size=self.total_channel_count * (scl_features),
            )
        if self.fcm_context_length > 0:
            patch_config = copy.deepcopy(config)
            patch_config.context_length = config.prediction_length + (2 * config.fcm_context_length)
            patch_config.masked_context_length = None
            patch_config.patch_length = self.scl
            patch_config.patch_stride = 1
            self.fcm_patch_block = TinyTimeMixerPatchify(patch_config)

        self.fcm_gated_attn = config.fcm_gated_attn
        self.prediction_length = config.prediction_length
        self.fcm_prepend_past = config.fcm_prepend_past

        self.fcm_prepend_past_offset = (
            config.fcm_prepend_past_offset
        )  # Number of items to skip in the context window from the end

        if self.fcm_prepend_past_offset is None:
            self.fcm_prepend_slicing_indices = slice(-self.fcm_context_length, None)
        else:
            self.fcm_prepend_slicing_indices = slice(
                -(self.fcm_prepend_past_offset + self.fcm_context_length),
                -self.fcm_prepend_past_offset,
            )

    def forward(
        self,
        base_forecasts: torch.Tensor,
        past_values: Optional[torch.Tensor],
        future_values: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            base_forecasts (`torch.Tensor` of shape `(batch_size, prediction length, forecast_channels)`):
                Base Forecasts to reconcile

            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

            future_values (`torch.Tensor` of shape `(batch_size, prediction length, input_channels)`, *optional*, Defaults to None):
                Actual groundtruths of the forecasts. Pass dummy values (say 0) for forecast channels, if groundtruth is unknown.
                Pass the correct values for Exogenous channels where the forecast values are known.

        Returns:
            `torch.Tensor`: Updated forecasts of shape `(batch_size, prediction length, forecast_channels)`
        """
        # base_forecasts.shape == (batch_size x forecast_len x n_vars)

        if self.prediction_channel_indices is not None:
            past_prepend_values = past_values[
                :, self.fcm_prepend_slicing_indices, self.prediction_channel_indices
            ]  # bs x context_len x forecast_channels
        else:
            past_prepend_values = past_values[
                :, self.fcm_prepend_slicing_indices, :
            ]  # bs x fcm_context_len x forecast_channels

        if self.exogenous_channel_count > 0 and future_values is None:
            raise ValueError("future_values cannot be none when we have exogenous channels.")

        if self.exogenous_channel_count > 0:
            exog_values = future_values[..., self.exogenous_channel_indices]  # bs x prediction len x exog_channels
            past_exog_values = past_values[
                :, self.fcm_prepend_slicing_indices, self.exogenous_channel_indices
            ]  # bs x context_len x exog_channels

            past_prepend_values = torch.cat(
                (past_prepend_values, past_exog_values), dim=-1
            )  # bs x fcm_context_len x (forecast_channels+exog_channels)

        else:
            exog_values = None

        residual = base_forecasts

        if exog_values is not None:
            base_forecasts = torch.cat(
                (base_forecasts, exog_values), dim=-1
            )  # x.shape == (batch_size x forecast_len x (forecast_channels+exog_channels))

        if self.fcm_context_length > 0:
            # this mode creates a patch for every multi-variate forecast point with its surronding context
            # it then flattens it and applies MLP to it.
            # By this way, every forecast point learn from its pre and post surrounding context in a channel mixed way.
            # Residual is added to ensure noise reduction with initial forecasts.

            # prefill and postfill zeros to enable patching for every forecast point with surrounding context

            dummy = torch.zeros(
                base_forecasts.shape[0],
                self.fcm_context_length,
                base_forecasts.shape[2],
                device=base_forecasts.device,
            )  # bs x fcm_context_length x n_vars

            if self.fcm_prepend_past:
                # add prefill and postfill
                extend_forecasts = torch.concat(
                    (past_prepend_values, base_forecasts, dummy), dim=1
                )  # bs x forecast_len + 2*fcm_context_length x n_vars
            else:
                # add prefill and postfill
                extend_forecasts = torch.concat(
                    (dummy, base_forecasts, dummy), dim=1
                )  # bs x forecast_len + 2*fcm_context_length x n_vars

            # create patch
            extend_forecasts = self.fcm_patch_block(extend_forecasts)  # xb: [bs x n_vars x forecast_len  x scl]

            extend_forecasts = extend_forecasts.transpose(1, 2)  # [bs x forecast_len  x n_vars  x scl]

            if extend_forecasts.shape[1] != self.prediction_length:
                raise ValueError("out_patches should match to forecast length")

            if self.fcm_use_mixer:
                extend_forecasts = self.fcm_embedding(extend_forecasts)
                extend_forecasts, _ = self.exog_mixer(extend_forecasts)

            extend_forecasts = extend_forecasts.flatten(start_dim=2)  # xb: [bs x forecast_len x n_vars * scl]

            if self.fcm_gated_attn:
                extend_forecasts = self.fcm_gating_block(extend_forecasts)  # xb: [bs x forecast_len x n_vars * scl]

            extend_forecasts = self.mlp(extend_forecasts)  # xb: [bs x forecast_len x n_vars]

        else:
            if self.fcm_gated_attn:
                extend_forecasts = self.fcm_gating_block(base_forecasts)

            extend_forecasts = self.mlp(extend_forecasts)

        new_forecast = extend_forecasts + residual

        return new_forecast


class TinyTimeMixerLayer(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.num_patches = config.num_patches
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TinyTimeMixerChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class TinyTimeMixerAdaptivePatchingBlock(nn.Module):
    """
    The `TinyTimeMixer` layer that does all three kinds of mixing.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TinyTimeMixerConfig, adapt_patch_level: int):
        super().__init__()
        temp_config = copy.deepcopy(config)
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor

        if config.d_model // self.adaptive_patch_factor <= 4:
            # do not allow reduction beyond d_model less than 4
            logger.warning(
                "Disabling adaptive patching at level %s. Either increase d_model or reduce adaptive_patching_levels"
                % (adapt_patch_level)
            )
            self.adaptive_patch_factor = 1

        if config.d_model % self.adaptive_patch_factor != 0:
            raise ValueError("d_model should be divisible by 2^i, where i varies from 0 to adaptive_patching_levels.")
        temp_config.num_patches = temp_config.num_patches * self.adaptive_patch_factor
        temp_config.d_model = temp_config.d_model // self.adaptive_patch_factor

        self.mixer_layers = nn.ModuleList([TinyTimeMixerLayer(temp_config) for i in range(temp_config.num_layers)])

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        all_hidden_states = []
        all_hidden_states.append(hidden)
        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] * self.adaptive_patch_factor,
                hidden.shape[3] // self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        for mod in self.mixer_layers:
            hidden = mod(hidden)
            all_hidden_states.append(hidden)

        hidden = torch.reshape(
            hidden,
            (
                hidden.shape[0],
                hidden.shape[1],
                hidden.shape[2] // self.adaptive_patch_factor,
                hidden.shape[3] * self.adaptive_patch_factor,
            ),
        )
        all_hidden_states.append(hidden)

        return hidden, all_hidden_states


class TinyTimeMixerBlock(nn.Module):
    """The main computing framework of the `TinyTimeMixer` model.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        self.adaptive_patching_levels = config.adaptive_patching_levels

        if self.adaptive_patching_levels > 0:
            self.mixers = nn.ModuleList(
                [
                    TinyTimeMixerAdaptivePatchingBlock(config=config, adapt_patch_level=i)
                    for i in reversed(range(config.adaptive_patching_levels))
                ]
            )

        else:
            self.mixers = nn.ModuleList([TinyTimeMixerLayer(config=config) for _ in range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            if self.adaptive_patching_levels > 0:
                embedding, hidden_states = mod(embedding)
                all_hidden_states.extend(hidden_states)
            else:
                embedding = mod(embedding)
                if output_hidden_states:
                    all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TinyTimeMixerDecoder(nn.Module):
    """Decoder for tiny time mixer

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        if config.d_model != config.decoder_d_model:
            self.adapter = nn.Linear(config.d_model, config.decoder_d_model)
        else:
            self.adapter = None

        self.decoder_raw_residual = config.decoder_raw_residual
        self.num_input_channels = config.num_input_channels

        if config.decoder_raw_residual:
            self.decoder_raw_embedding = nn.Linear(config.patch_length, config.decoder_d_model)
            # nn.init.zeros_(self.decoder_raw_embedding.weight)
            # nn.init.zeros_(self.decoder_raw_embedding.bias)

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.decoder_d_model
        decoder_config.dropout = config.head_dropout
        decoder_config.adaptive_patching_levels = config.decoder_adaptive_patching_levels
        decoder_config.mode = config.decoder_mode

        if config.categorical_vocab_size_list is not None:
            if config.decoder_mode == "common_channel":
                # logger.warning("Setting decoder_mode to mix_channel as static categorical variables is available")
                # config.decoder_mode = "mix_channel"
                raise ValueError("set decoder_mode to mix_channel when using static categorical variables")

            decoder_config.num_input_channels += len(config.categorical_vocab_size_list)
            self.decoder_cat_embedding_layer = TinyTimeMixerCategoricalEmbeddingLayer(decoder_config)
        else:
            self.decoder_cat_embedding_layer = None

        self.decoder_block = TinyTimeMixerBlock(decoder_config)

        self.resolution_prefix_tuning = config.resolution_prefix_tuning

    def forward(
        self,
        hidden_state,
        patch_input,
        output_hidden_states: bool = False,
        static_categorical_values: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`): The input tensor from backbone.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

            static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        if output_hidden_states:
            decoder_hidden_states = []
        else:
            decoder_hidden_states = None

        decoder_input = hidden_state

        if self.adapter is not None:
            decoder_input = self.adapter(
                hidden_state
            )  # model_output: [batch_size x nvars x num_patch x decoder_d_model]
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        if self.decoder_raw_residual:
            if self.resolution_prefix_tuning:
                if patch_input.shape[-2] == decoder_input.shape[-2] - 1:
                    temp_shape = list(patch_input.shape)
                    temp_shape[-2] = 1
                    temp_zeros = torch.zeros(*temp_shape).to(patch_input.device)
                    patch_input = torch.cat([temp_zeros, patch_input], dim=-2)

            decoder_input = decoder_input + self.decoder_raw_embedding(
                patch_input
            )  # [batch_size x nvars x num_patch x decoder_d_model]
            if output_hidden_states:
                decoder_hidden_states.append(decoder_input)

        if self.decoder_cat_embedding_layer is not None:
            if static_categorical_values is None:
                raise ValueError("Missing static_categorical_values tensor in forward call")
            cat_embeddings = self.decoder_cat_embedding_layer(
                static_categorical_values
            )  # bs x n_cat x n_patches x d_model

            decoder_input = torch.concat(
                (decoder_input, cat_embeddings), dim=1
            )  # bs x nvars+n_cat x n_patches x d_model

        decoder_output, hidden_states = self.decoder_block(
            hidden_state=decoder_input, output_hidden_states=output_hidden_states
        )  # bs x nvars+n_cat x n_patches x d_model

        if output_hidden_states:
            decoder_hidden_states.extend(hidden_states)

        if self.decoder_cat_embedding_layer is not None:
            decoder_output = decoder_output[:, : self.num_input_channels, :, :]  # bs x nvars x n_patches x d_model
            if output_hidden_states:
                decoder_hidden_states.append(decoder_output)

        return decoder_output, decoder_hidden_states


class TinyTimeMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`TinyTimeMixerConfig`, *required*): Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices = sorted(self.prediction_channel_indices)

        self.prediction_filter_length = config.prediction_filter_length

        self.dropout_layer = nn.Dropout(config.head_dropout)
        self.enable_forecast_channel_mixing = config.enable_forecast_channel_mixing
        if config.use_decoder:
            head_d_model = config.decoder_d_model
        else:
            head_d_model = config.d_model

        if distribution_output is None:
            self.base_forecast_block = nn.Linear((config.num_patches * head_d_model), config.prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(config.num_patches * head_d_model)

        self.flatten = nn.Flatten(start_dim=-2)

        if self.enable_forecast_channel_mixing:
            temp_config = copy.deepcopy(config)
            if self.prediction_filter_length is not None:
                temp_config.prediction_length = self.prediction_filter_length

            self.fcm_block = ForecastChannelHeadMixer(config=temp_config)

    def forward(self, hidden_features, past_values, future_values=None):
        """

        Args:
            hidden_features `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a forecasting task, this denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

            future_values (`torch.Tensor` of shape `(batch_size, prediction length, input_channels)`, *optional*, Defaults to None):
                Actual groundtruths of the forecasts. Pass dummy values (say 0) for forecast channels, if groundtruth is unknown.
                Pass the correct values for Exogenous channels where the forecast values are known.


        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, forecast_channels)`.

        """

        hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        forecast = self.base_forecast_block(hidden_features)  # [batch_size x n_vars x prediction_length]
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)  # [batch_size x prediction_length x n_vars]

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)
            else:
                forecast = forecast[
                    ..., self.prediction_channel_indices
                ]  # [batch_size x prediction_length x prediction_n_vars]

        if self.prediction_filter_length is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[:, : self.prediction_filter_length, :] for z in forecast)
            else:
                forecast = forecast[
                    :, : self.prediction_filter_length, :
                ]  # [batch_size x prediction_filter_length x prediction_n_vars]

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[
                :, : self.prediction_filter_length, :
            ]  # [batch_size x prediction_filter_length x n_vars]

        if self.enable_forecast_channel_mixing:
            if isinstance(forecast, tuple):
                raise ValueError("Forecast channel mixing is not enabled for distribution head")
            else:
                forecast = self.fcm_block(forecast, past_values=past_values, future_values=future_values)
                # [batch_size x prediction_length x prediction_n_vars]

        return forecast


class TinyTimeMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = TinyTimeMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""

        # if isinstance(module, MultiQuantileHead):
        #     # Initialize MQ temperature parameter if enabled.
        #     # This avoids doing init inside the head and keeps HF-style init central.
        #     if (
        #         getattr(module, "enable_delta_temperature", False)
        #         and module._temp_u is not None
        #     ):
        #         init_temp = float(getattr(self.config, "mq_temperature_init", 1.0))
        #         init_temp = max(init_temp, 1e-8)

        #         # softplus^{-1}(x) = log(exp(x) - 1)
        #         init_u = torch.log(torch.expm1(torch.tensor(init_temp)))

        #         with torch.no_grad():
        #             module._temp_u.data.fill_(init_u)
        if isinstance(module, TinyTimeMixerPositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TinyTimeMixerBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            # print(f"Initializing Linear layers with method: {self.config.init_linear}")
            if self.config.init_linear == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif self.config.init_linear == "uniform":
                nn.init.uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif self.config.init_linear == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            # print(f"Initializing Embedding layers with method: {self.config.init_embed}")
            if self.config.init_embed == "normal":
                nn.init.normal_(module.weight)
            elif self.config.init_embed == "uniform":
                nn.init.uniform_(module.weight)
            elif self.config.init_embed == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            else:
                module.reset_parameters()
        elif isinstance(module, nn.Conv1d):
            nn.init.constant_(module.weight, 1.0 / module.kernel_size[0])


class TinyTimeMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.sequence_length = (
            config.masked_context_length if config.masked_context_length is not None else config.context_length
        )

        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


class TinyTimeMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """

        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(torch.tensor(1, device=denominator.device))
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        normalized = self.transform(data, loc, scale)
        return normalized, loc, scale

    def transform(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """
        normalized = (data - loc) / scale

        # if self.suppress_outliers:
        #     normalized = torch.arcsinh(normalized)

        return normalized

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """

        # if self.suppress_outliers:
        #     data = torch.sinh(data)

        restored = data * scale + loc
        # restored = torch.where(observed_indicator.bool(), restored, data)
        return restored


class TinyTimeMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale

    def transform(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """
        normalized = data / scale

        # if self.suppress_outliers:
        #     normalized = torch.arcsinh(normalized)

        return normalized

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """
        restored = data * scale + loc
        # restored = torch.where(observed_indicator.bool(), restored, data)
        return restored


class TinyTimeMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale

    def transform(
        self,
        data: torch.Tensor,
        loc: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """ """
        return data

    def inverse(
        self,
        data: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization + affine transform on observed values.
        """
        restored = data * scale + loc
        # restored = torch.where(observed_indicator.bool(), restored, data)
        return restored


@dataclass
class TinyTimeMixerEncoderOutput(ModelOutput):
    """
    Base class for `TinyTimeMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MultiScaleFromPatchedSequence(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig):
        super().__init__()

        self.patch_len = config.patch_length
        self.d_model = config.d_model
        max_seq_len = config.context_length

        self.max_scales = 0
        seq_len = max_seq_len
        while seq_len >= self.patch_len:
            self.max_scales += 1
            seq_len = seq_len // 2

        self.projectors = nn.ModuleList([nn.Linear(self.patch_len, self.d_model) for _ in range(self.max_scales)])

    def forward(self, patched_x):
        B, C, P, L = patched_x.shape
        S = P * L
        x = patched_x.view(B, C, S)

        outputs = []
        # downsampled_sequences = []
        # num_patches_per_scale = []

        i = 0
        while True:
            factor = 2**i
            if S < factor or S // factor < self.patch_len:
                break

            downsampled = F.avg_pool1d(x, kernel_size=factor, stride=factor)
            # downsampled_sequences.append((i, downsampled[0, 0].tolist()))

            S_i = downsampled.shape[-1]
            num_patch_i = S_i // self.patch_len
            if num_patch_i == 0:
                break

            downsampled = downsampled[:, :, -num_patch_i * self.patch_len :]
            patches = downsampled.reshape(B, C, num_patch_i, self.patch_len)

            projected = self.projectors[i](patches.reshape(B * C * num_patch_i, self.patch_len))
            projected = projected.reshape(B, C, num_patch_i, self.d_model)
            # summed = projected.sum(dim=2)
            outputs.append(projected)
            # num_patches_per_scale.append(num_patch_i)

            i += 1

        outputs.reverse()  # add hierarchies features to end
        return torch.cat(outputs, dim=2)
        # num_patches_per_scale, downsampled_sequences


class TinyTimeMixerAddLearnableRegisterTokens(nn.Module):
    def __init__(self, config: TinyTimeMixerConfig, device):
        super(TinyTimeMixerAddLearnableRegisterTokens, self).__init__()
        self.register_tokens = config.register_tokens
        d_model = config.d_model

        self.patch_tokens = None
        # Learnable patch tokens (p): shape (num_patch_tokens x d_model)
        if self.register_tokens > 0:
            self.patch_tokens = nn.Parameter(torch.randn(self.register_tokens, d_model).to(device))

    def forward(self, x, patch_mask=None):
        # Input x shape: batch x num_channels x num_patches x d_model
        batch_size, num_channels, num_patches, d_model = x.size()

        if self.patch_tokens is not None:
            # Expand patch tokens along the batch and channel dimensions
            # Result shape: (1 x 1 x num_patch_tokens x d_model)
            patch_tokens_expanded = self.patch_tokens.unsqueeze(0).unsqueeze(0)

            # Add patch tokens to the num_patches dimension
            # Shape: (batch x num_channels x (num_patches + num_patch_tokens) x d_model)
            x = torch.cat(
                [patch_tokens_expanded.expand(batch_size, num_channels, -1, -1), x],
                dim=2,
            )
            patch_mask = update_patch_mask(patch_mask, self.register_tokens)

        return x, patch_mask


class TinyTimeMixerAddFFTPatches(nn.Module):
    """
    Backward-compatible FFT token injection.

    Default behavior (backward compatible):
      - get_one_freq_emb = False
      - fft_ignore_dc    = False
      - Adds k FFT tokens (one per selected freq id), each token has dim = d_model
        x: [B, C, P, D] -> [B, C, P+k, D]

    Optional behavior (get_one_freq_emb = True):
      - Embeds each of the top-k freq ids into d_freq (auto-chosen)
      - Concats k*d_freq and projects to d_model to form ONE token
      - Adds only 1 token:
        x: [B, C, P, D] -> [B, C, P+1, D]

    Config params used:
      - config.fft_length (int)
      - config.d_model (int)
      - config.context_length (int)
      - config.use_fft_embedding (bool, default False)
      - config.get_one_freq_emb (bool, default False)

      - config.fft_ignore_dc (bool, default False)   # keep exact old behavior by default

      - config.fft_d_freq_min (int, default 4)       # only used when get_one_freq_emb=True
      - config.fft_d_freq_max (int, default 64)      # only used when get_one_freq_emb=True
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d_model = int(config.d_model)

        self.seq_len = int(config.context_length)
        self.max_freq_bins = self.seq_len // 2 + 1  # rfft output length

        # Cap fft_k to avoid torch.topk crash
        self.fft_k = min(int(getattr(config, "fft_length", 4)), self.max_freq_bins)

        self.use_fft_embedding = bool(getattr(config, "use_fft_embedding", False))

        # New flag (default False): keep old behavior unless enabled
        self.get_one_freq_emb = bool(getattr(config, "get_one_freq_emb", False))

        # Backward-compatible default: do NOT ignore DC unless enabled
        self.fft_ignore_dc = bool(getattr(config, "fft_ignore_dc", False))

        # ----- Build embedding modules -----
        if self.get_one_freq_emb:
            # Auto d_freq based on fft_k and d_model, but clamp to meaningful range
            raw = max(1, self.d_model // max(1, self.fft_k))

            d_freq_min = int(getattr(config, "fft_d_freq_min", 4))
            d_freq_max = int(getattr(config, "fft_d_freq_max", 64))
            self.d_freq = int(max(d_freq_min, min(d_freq_max, raw)))

            if self.use_fft_embedding:
                self.freq_embedding = nn.Embedding(self.max_freq_bins, self.d_freq)
            else:
                self.freq_index_mlp = nn.Sequential(
                    nn.Linear(1, self.d_freq),
                    nn.ReLU(),
                    nn.Linear(self.d_freq, self.d_freq),
                )

            # Concat k embeddings (k*d_freq) -> project to d_model (one token)
            self.freq_concat_proj = nn.Linear(self.fft_k * self.d_freq, self.d_model)

        else:
            # Original behavior: each of k tokens is d_model-sized
            self.d_freq = self.d_model

            if self.use_fft_embedding:
                self.freq_embedding = nn.Embedding(self.max_freq_bins, self.d_model)
            else:
                self.freq_index_mlp = nn.Sequential(
                    nn.Linear(1, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.d_model),
                )

    def forward(self, x, raw_input, patch_mask=None):
        """
        Args:
            x: [B, C, P, D] patched tokens
            raw_input: [B, S, C] original time-series (or longer; we slice to context_length)
            patch_mask: optional mask corresponding to x's patch dimension

        Returns:
            x: updated tokens
            patch_mask: updated mask (if provided)
        """
        # slice for maskedprediction workflow
        raw_input = raw_input[:, : self.seq_len, :]  # [B, S, C]
        B, S, C = raw_input.shape

        # FFT: [B, F, C] where F = S//2 + 1
        fft = torch.fft.rfft(raw_input, dim=1)
        mag = fft.abs()  # [B, F, C]

        # Keep backward compatibility by default (fft_ignore_dc=False)
        if self.fft_ignore_dc:
            mag[:, 0, :] = 0.0

        # Top-k selection along frequency dimension
        # (fft_k was already capped to max_freq_bins in __init__)
        topk = torch.topk(mag, self.fft_k, dim=1)
        topk_indices = topk.indices.permute(0, 2, 1)  # [B, C, k]

        # Embed frequency ids -> [B, C, k, d_*]
        if self.use_fft_embedding:
            freq_tokens = self.freq_embedding(topk_indices)
        else:
            norm_bin_indices = topk_indices.float() / float(self.max_freq_bins)  # [B, C, k]
            freq_tokens = self.freq_index_mlp(norm_bin_indices.unsqueeze(-1))

        if not self.get_one_freq_emb:
            # Original behavior: prepend k tokens
            x = torch.cat([freq_tokens, x], dim=2)  # [B, C, P+k, D]
            if patch_mask is not None:
                patch_mask = update_patch_mask(patch_mask, self.fft_k)
            return x, patch_mask

        # New behavior: build one token from top-k embeddings
        # freq_tokens: [B, C, k, d_freq] -> [B, C, k*d_freq] -> proj -> [B, C, 1, d_model]
        freq_flat = freq_tokens.reshape(B, C, self.fft_k * self.d_freq)
        one_token = self.freq_concat_proj(freq_flat).unsqueeze(2)

        x = torch.cat([one_token, x], dim=2)  # [B, C, P+1, D]
        if patch_mask is not None:
            patch_mask = update_patch_mask(patch_mask, 1)
        return x, patch_mask


DEFAULT_Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _is_default_9_quantiles(q_list, tol=1e-12):
    if len(q_list) != 9:
        return False
    for q1, q2 in zip(q_list, DEFAULT_Q):
        if abs(q1 - q2) > tol:
            return False
    return True


class MultiQuantileHead(nn.Module):
    """
    Generic multi-quantile head.

    - Uses config.quantile_levels (default: [0.1..0.9]) instead of hardcoded 9 quantiles.
    - Assumes:
        * quantile_levels contains 0.5 (median)
        * len(quantile_levels) is odd
    - For the default quantile_levels = [0.1,0.2,...,0.9], this produces *exactly*
      the same tensor layout/values as the original hardcoded implementation.
    """

    def __init__(self, config):
        super().__init__()

        self.mq_hidden = int(getattr(config, "mq_hidden", 8))
        self.mq_kernel_size = int(getattr(config, "mq_kernel_size", 3))
        self.mq_eps = float(getattr(config, "mq_eps", 1e-6))

        self.decoder_d_model = int(config.decoder_d_model)
        self.num_patches = int(config.num_patches)  # ← use backbone config

        self.mq_use_decoder_pool = bool(getattr(config, "mq_use_decoder_pool", False))
        self.mq_cond_path = str(getattr(config, "mq_cond_path", "pool")).lower()
        self.mq_cond_mode = str(getattr(config, "mq_cond_mode", "add")).lower()

        assert self.mq_cond_path in ("pool", "flatten")
        assert self.mq_cond_mode in ("add", "concat")

        self.mq_q50_type = str(getattr(config, "mq_q50_type", "median")).lower()

        # -----------------------------
        # Quantile list (generic)
        # -----------------------------
        qlist = getattr(config, "quantile_levels", None)
        if qlist is None:
            qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # normalize + sort
        qlist = [float(q) for q in qlist]
        qlist = sorted(qlist)

        Q = len(qlist)
        if Q % 2 != 1:
            raise ValueError(f"quantile_levels must be odd length, got {Q}: {qlist}")
        # median must exist
        # NOTE: float comparisons can be finicky; use a small tolerance
        median_idx = None
        for i, q in enumerate(qlist):
            if abs(q - 0.5) < 1e-12:
                median_idx = i
                break
        if median_idx is None:
            raise ValueError(f"quantile_levels must contain 0.5 (median). Got: {qlist}")

        # we assume odd -> symmetric counts around median index
        k_down = median_idx
        k_up = Q - median_idx - 1
        if k_down != k_up:
            # You said we can assume odd+median; if you also always want symmetry,
            # keep this check. If you want to allow non-symmetric lists, remove it.
            raise ValueError(
                f"quantile_levels must have equal counts below/above median. "
                f"Got k_down={k_down}, k_up={k_up}, list={qlist}"
            )

        self.quantile_list = qlist
        self.num_quantiles = Q
        self.median_index = median_idx
        self.k_side = k_down  # number below (and above) median

        # --- base stack ---
        pad = self.mq_kernel_size // 2
        self.dw = nn.Conv1d(1, 1, kernel_size=self.mq_kernel_size, padding=pad)
        self.inorm = nn.InstanceNorm1d(1, affine=True)
        self.pw1 = nn.Conv1d(1, self.mq_hidden, kernel_size=1)
        self.act = nn.GELU()

        # --- decoder conditioning ---
        if self.mq_use_decoder_pool:
            if self.mq_cond_path == "pool":
                self.dec_proj = nn.Linear(self.decoder_d_model, self.mq_hidden)
            else:  # flatten path
                self.mq_decoder_d_model = int(getattr(config, "mq_decoder_d_model", self.mq_hidden))

                # D → md (token-wise)
                self.dec_tok_proj = nn.Linear(self.decoder_d_model, self.mq_decoder_d_model)

                # (num_patches * md) → hidden
                flat_dim = self.num_patches * self.mq_decoder_d_model
                self.dec_flat_to_hidden = nn.Linear(flat_dim, self.mq_hidden)

        # adjust pw2 for concat
        pw2_in = self.mq_hidden
        if self.mq_use_decoder_pool and self.mq_cond_mode == "concat":
            pw2_in = self.mq_hidden * 2

        # IMPORTANT: out_channels is now generic (Q)
        self.pw2 = nn.Conv1d(pw2_in, self.num_quantiles, kernel_size=1)

    def forward(self, mean_hat, decoder_hidden_state=None):
        # mean_hat: [B,T,C]
        mean_hat_ct = mean_hat.transpose(-1, -2).contiguous()  # [B,C,T]
        B, C, T = mean_hat_ct.shape

        x = mean_hat_ct.view(B * C, 1, T)
        x = self.inorm(self.dw(x))
        h = self.act(self.pw1(x))  # [B*C,H,T]

        if self.mq_use_decoder_pool:
            if decoder_hidden_state is None:
                raise ValueError("decoder_hidden_state required")

            if self.mq_cond_path == "pool":
                dec_pool = decoder_hidden_state.mean(dim=2)  # [B,C,D]
                cond = self.dec_proj(dec_pool)  # [B,C,H]
            else:  # flatten path
                B2, C2, P, D = decoder_hidden_state.shape
                if P != self.num_patches:
                    raise ValueError(f"Expected {self.num_patches} patches, got {P}")

                dec_tok = self.dec_tok_proj(decoder_hidden_state)  # [B,C,P,md]
                dec_flat = dec_tok.reshape(B2, C2, -1)  # [B,C,P*md]
                cond = self.dec_flat_to_hidden(dec_flat)  # [B,C,H]

            cond = cond.view(B * C, -1).unsqueeze(-1).expand(-1, -1, T)  # [B*C,H,T]

            if self.mq_cond_mode == "add":
                h = h + cond
            else:
                h = torch.cat([h, cond], dim=1)

        # o: [B,C,Q,T]
        o = self.pw2(h).view(B, C, self.num_quantiles, T)

        # Channel layout matches the old behavior when Q=9:
        #   o[:, :, 0, :]      -> bias50
        #   o[:, :, 1:1+k, :]  -> down_raw (nearest->farthest below median)
        #   o[:, :, 1+k:, :]   -> up_raw   (nearest->farthest above median)
        k = self.k_side
        bias50 = o[:, :, 0:1, :]
        down_raw = o[:, :, 1 : 1 + k, :]
        up_raw = o[:, :, 1 + k : 1 + 2 * k, :]

        down_inc = F.softplus(down_raw) + self.mq_eps
        up_inc = F.softplus(up_raw) + self.mq_eps

        # median anchor
        q50 = mean_hat_ct.unsqueeze(2)  # [B,C,1,T]
        if self.mq_q50_type == "median":
            q50 = q50 + bias50

        # cumulative distances (nearest -> farthest)
        down_cum = torch.cumsum(down_inc, dim=2)  # [B,C,k,T]
        up_cum = torch.cumsum(up_inc, dim=2)  # [B,C,k,T]

        if _is_default_9_quantiles(self.quantile_list):
            # hardcoded identical graph
            q40 = q50 - down_cum[:, :, 0:1, :]
            q30 = q50 - down_cum[:, :, 1:2, :]
            q20 = q50 - down_cum[:, :, 2:3, :]
            q10 = q50 - down_cum[:, :, 3:4, :]

            q60 = q50 + up_cum[:, :, 0:1, :]
            q70 = q50 + up_cum[:, :, 1:2, :]
            q80 = q50 + up_cum[:, :, 2:3, :]
            q90 = q50 + up_cum[:, :, 3:4, :]

            qhat_ctqt = torch.cat([q10, q20, q30, q40, q50, q60, q70, q80, q90], dim=2)
        else:
            # generic flip-based path
            down_near_to_far = q50 - down_cum
            up_near_to_far = q50 + up_cum
            down_asc = torch.flip(down_near_to_far, dims=[2])
            qhat_ctqt = torch.cat([down_asc, q50, up_near_to_far], dim=2)
        # # quantiles around median
        # # down_near_to_far: [q_(median-1), q_(median-2), ..., q_(lowest)] in that order
        # down_near_to_far = q50 - down_cum
        # up_near_to_far = q50 + up_cum

        # # Assemble in ascending quantile order:
        # #   [lowest ... just-below-median, median, just-above-median ... highest]
        # # Old code did: [q10,q20,q30,q40,q50,q60,q70,q80,q90]
        # down_asc = torch.flip(
        #     down_near_to_far, dims=[2]
        # )  # farthest->nearest => ascending tail below median
        # qhat_ctqt = torch.cat([down_asc, q50, up_near_to_far], dim=2)  # [B,C,Q,T]

        # return: [B,Q,T,C]
        return qhat_ctqt.permute(0, 2, 3, 1).contiguous()


# class MultiQuantileHead(nn.Module):

#     def __init__(self, config):
#         super().__init__()

#         self.mq_hidden = int(getattr(config, "mq_hidden", 8))
#         self.mq_kernel_size = int(getattr(config, "mq_kernel_size", 3))
#         self.mq_eps = float(getattr(config, "mq_eps", 1e-6))

#         self.decoder_d_model = int(config.decoder_d_model)
#         self.num_patches = int(config.num_patches)  # ← use backbone config

#         self.mq_use_decoder_pool = bool(getattr(config, "mq_use_decoder_pool", False))
#         self.mq_cond_path = str(getattr(config, "mq_cond_path", "pool")).lower()
#         self.mq_cond_mode = str(getattr(config, "mq_cond_mode", "add")).lower()

#         assert self.mq_cond_path in ("pool", "flatten")
#         assert self.mq_cond_mode in ("add", "concat")

#         self.mq_q50_type = str(getattr(config, "mq_q50_type", "median")).lower()

#         # --- base stack ---
#         pad = self.mq_kernel_size // 2
#         self.dw = nn.Conv1d(1, 1, kernel_size=self.mq_kernel_size, padding=pad)
#         self.inorm = nn.InstanceNorm1d(1, affine=True)
#         self.pw1 = nn.Conv1d(1, self.mq_hidden, kernel_size=1)
#         self.act = nn.GELU()

#         # --- decoder conditioning ---
#         if self.mq_use_decoder_pool:

#             if self.mq_cond_path == "pool":
#                 self.dec_proj = nn.Linear(self.decoder_d_model, self.mq_hidden)

#             else:  # flatten path

#                 self.mq_decoder_d_model = int(
#                     getattr(config, "mq_decoder_d_model", self.mq_hidden)
#                 )

#                 # D → md (token-wise)
#                 self.dec_tok_proj = nn.Linear(
#                     self.decoder_d_model,
#                     self.mq_decoder_d_model,
#                 )

#                 # (num_patches * md) → hidden
#                 flat_dim = self.num_patches * self.mq_decoder_d_model
#                 self.dec_flat_to_hidden = nn.Linear(flat_dim, self.mq_hidden)

#         # adjust pw2 for concat
#         pw2_in = self.mq_hidden
#         if self.mq_use_decoder_pool and self.mq_cond_mode == "concat":
#             pw2_in = self.mq_hidden * 2

#         self.pw2 = nn.Conv1d(pw2_in, 9, kernel_size=1)

#     def forward(self, mean_hat, decoder_hidden_state=None):

#         mean_hat_ct = mean_hat.transpose(-1, -2).contiguous()
#         B, C, T = mean_hat_ct.shape

#         x = mean_hat_ct.view(B * C, 1, T)
#         x = self.inorm(self.dw(x))
#         h = self.act(self.pw1(x))  # [B*C,H,T]

#         if self.mq_use_decoder_pool:

#             if decoder_hidden_state is None:
#                 raise ValueError("decoder_hidden_state required")

#             if self.mq_cond_path == "pool":

#                 dec_pool = decoder_hidden_state.mean(dim=2)  # [B,C,D]
#                 cond = self.dec_proj(dec_pool)

#             else:  # flatten path

#                 B2, C2, P, D = decoder_hidden_state.shape
#                 assert (
#                     P == self.num_patches
#                 ), f"Expected {self.num_patches} patches, got {P}"

#                 # [B,C,P,D] → [B,C,P,md]
#                 dec_tok = self.dec_tok_proj(decoder_hidden_state)

#                 # flatten patches
#                 dec_flat = dec_tok.reshape(B2, C2, -1)

#                 # project to hidden
#                 cond = self.dec_flat_to_hidden(dec_flat)  # [B,C,H]

#             cond = cond.view(B * C, -1).unsqueeze(-1).expand(-1, -1, T)

#             if self.mq_cond_mode == "add":
#                 h = h + cond
#             else:
#                 h = torch.cat([h, cond], dim=1)

#         o = self.pw2(h).view(B, C, 9, T)

#         bias50 = o[:, :, 0:1, :]
#         down_raw = o[:, :, 1:5, :]
#         up_raw = o[:, :, 5:9, :]

#         down_inc = F.softplus(down_raw) + self.mq_eps
#         up_inc = F.softplus(up_raw) + self.mq_eps

#         q50 = mean_hat_ct.unsqueeze(2)
#         if self.mq_q50_type == "median":
#             q50 = q50 + bias50

#         down_cum = torch.cumsum(down_inc, dim=2)
#         up_cum = torch.cumsum(up_inc, dim=2)

#         q40 = q50 - down_cum[:, :, 0:1, :]
#         q30 = q50 - down_cum[:, :, 1:2, :]
#         q20 = q50 - down_cum[:, :, 2:3, :]
#         q10 = q50 - down_cum[:, :, 3:4, :]

#         q60 = q50 + up_cum[:, :, 0:1, :]
#         q70 = q50 + up_cum[:, :, 1:2, :]
#         q80 = q50 + up_cum[:, :, 2:3, :]
#         q90 = q50 + up_cum[:, :, 3:4, :]

#         qhat_ctqt = torch.cat([q10, q20, q30, q40, q50, q60, q70, q80, q90], dim=2)

#         return qhat_ctqt.permute(0, 2, 3, 1).contiguous()


# ----------------------------
# # Multi-Quantile Head (9 taus)
# # ----------------------------
# class MultiQuantileHeadOLD(nn.Module):
#     """
#     9-quantile head (0.1..0.9) with optional CRPS improvements.

#     Default behavior = IDENTICAL to your current implementation.
#     New behavior is enabled only if config.mq_* flags are set.

#     Input:  mean_hat [B, T, C]
#     Output: qhat     [B, 9, T, C]  for taus = 0.1..0.9
#     """

#     def __init__(self, config):
#         super().__init__()

#         # --- Base params (same as before) ---
#         hidden = int(getattr(config, "mq_hidden", 8))
#         kernel_size = int(getattr(config, "mq_kernel_size", 3))
#         self.eps = float(getattr(config, "mq_eps", 1e-6))
#         pad = kernel_size // 2

#         # --- Optional CRPS flags ---
#         self.detach_mean_for_head = bool(
#             getattr(config, "mq_detach_mean_for_head", False)
#         )
#         self.median_mode = str(getattr(config, "mq_median_mode", "biased"))
#         self.median_bias_shrink = float(getattr(config, "mq_median_bias_shrink", 0.05))

#         self.enable_delta_temperature = bool(
#             getattr(config, "mq_enable_delta_temperature", False)
#         )
#         self.temperature_per_horizon = bool(
#             getattr(config, "mq_temperature_per_horizon", False)
#         )

#         self.prediction_length = int(getattr(config, "prediction_length", 0))
#         if self.prediction_length <= 0:
#             raise ValueError(
#                 "config.prediction_length must be set (>0) for MultiQuantileHead."
#             )

#         if self.median_mode not in ("biased", "fixed", "shrink"):
#             raise ValueError(
#                 f"mq_median_mode must be one of ['biased','fixed','shrink'], got {self.median_mode}"
#             )

#         # --- Core layers (UNCHANGED from your baseline) ---
#         self.dw = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=True)
#         self.inorm = nn.InstanceNorm1d(1, affine=True)
#         self.pw1 = nn.Conv1d(1, hidden, kernel_size=1, bias=True)
#         self.pw2 = nn.Conv1d(
#             hidden, 9, kernel_size=1, bias=True
#         )  # [bias50, 4*down, 4*up]
#         self.act = nn.GELU()

#         # --- Optional temperature parameter (INIT done in model._init_weights) ---
#         self._temp_u = None
#         if self.enable_delta_temperature:
#             if self.temperature_per_horizon:
#                 self._temp_u = nn.Parameter(torch.zeros(self.prediction_length))
#             else:
#                 self._temp_u = nn.Parameter(torch.zeros(1))

#     def _get_temp(self, T: int, device, dtype):
#         """
#         Returns multiplicative temperature for deltas.
#         - scalar:       [1,1,1,1]
#         - per-horizon:  [1,1,1,T]
#         """
#         if not self.enable_delta_temperature:
#             return None
#         if self._temp_u is None:
#             return None

#         if self.temperature_per_horizon:
#             if T > self._temp_u.numel():
#                 raise ValueError(
#                     f"T={T} exceeds configured prediction_length={self._temp_u.numel()} for mq_temperature_per_horizon."
#                 )
#             temp = (
#                 F.softplus(self._temp_u[:T]).to(device=device, dtype=dtype) + self.eps
#             )
#             return temp.view(1, 1, 1, T)

#         temp = F.softplus(self._temp_u).to(device=device, dtype=dtype) + self.eps
#         return temp.view(1, 1, 1, 1)

#     def forward(self, mean_hat: torch.Tensor) -> torch.Tensor:
#         """
#         mean_hat: [B, T, C]
#         returns:  [B, 9, T, C]
#         """
#         if mean_hat.dim() != 3:
#             raise ValueError("mean_hat must be rank-3: [B, T, C].")

#         mean_hat_ct = mean_hat.transpose(-1, -2)  # [B, C, T]
#         B, C, T = mean_hat_ct.shape

#         # Optional: protect point predictor from quantile gradients
#         cond = mean_hat_ct.detach() if self.detach_mean_for_head else mean_hat_ct

#         x = cond.reshape(B * C, 1, T)
#         x = self.dw(x)
#         x = self.inorm(x)
#         h = self.act(self.pw1(x))
#         o = self.pw2(h).reshape(B, C, 9, T)

#         bias50 = o[:, :, 0:1, :]  # [B,C,1,T]
#         down_raw = o[:, :, 1:5, :]  # [B,C,4,T]
#         up_raw = o[:, :, 5:9, :]  # [B,C,4,T]

#         down_inc = F.softplus(down_raw) + self.eps
#         up_inc = F.softplus(up_raw) + self.eps

#         # Optional temperature scaling (starts as 1.0 if mq_temperature_init=1.0)
#         temp = self._get_temp(T, down_inc.device, down_inc.dtype)
#         if temp is not None:
#             down_inc = down_inc * temp
#             up_inc = up_inc * temp

#         # Median/anchor
#         if self.median_mode == "fixed":
#             q50 = mean_hat_ct.unsqueeze(2)  # MASE-safe anchor
#         elif self.median_mode == "shrink":
#             q50 = mean_hat_ct.unsqueeze(2) + self.median_bias_shrink * bias50
#         else:  # "biased" (baseline behavior)
#             q50 = mean_hat_ct.unsqueeze(2) + bias50

#         # cumulative distances from median
#         down_cum = torch.cumsum(down_inc, dim=2)
#         up_cum = torch.cumsum(up_inc, dim=2)

#         q40 = q50 - down_cum[:, :, 0:1, :]
#         q30 = q50 - down_cum[:, :, 1:2, :]
#         q20 = q50 - down_cum[:, :, 2:3, :]
#         q10 = q50 - down_cum[:, :, 3:4, :]

#         q60 = q50 + up_cum[:, :, 0:1, :]
#         q70 = q50 + up_cum[:, :, 1:2, :]
#         q80 = q50 + up_cum[:, :, 2:3, :]
#         q90 = q50 + up_cum[:, :, 3:4, :]

#         qhat_ctqt = torch.cat(
#             [q10, q20, q30, q40, q50, q60, q70, q80, q90], dim=2
#         )  # [B,C,9,T]
#         qhat = qhat_ctqt.permute(0, 2, 3, 1).contiguous()  # [B,9,T,C]
#         return qhat


# class MultiQuantileHead(nn.Module):
#     """
#     Turn mean forecasts [B, C, T] into quantiles [B, C, 9, T] for q = 0.1..0.9.
#     - Channel independent: same weights applied to every channel.
#     - Monotonic by construction using positive cumulative deltas (softplus).
#     - Deltas are conditioned on the mean via a small temporal conv stack.

#     Quantile layout (dim=2): [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#     """

#     def __init__(self, hidden: int = 8, kernel_size: int = 3, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         pad = kernel_size // 2

#         # Per-channel temporal features (shared across channels)
#         self.dw = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=pad, bias=True)
#         self.inorm = nn.InstanceNorm1d(1, affine=True)  # time-wise stabilization
#         # Lightweight projection to quantile params
#         self.pw1 = nn.Conv1d(1, hidden, kernel_size=1, bias=True)
#         self.pw2 = nn.Conv1d(
#             hidden, 9, kernel_size=1, bias=True
#         )  # [bias50, 4*down, 4*up]

#         self.act = nn.GELU()

#     def _safe_time_norm(self, x, eps=1e-3, clip=10.0):
#         # x: [B*C, 1, T]
#         mu = x.mean(dim=2, keepdim=True)
#         var = x.var(dim=2, unbiased=False, keepdim=True)
#         xn = (x - mu) / (var.add_(eps).sqrt_())
#         return xn.clamp_(-clip, clip)

#     def forward(self, mean_hat: torch.Tensor) -> torch.Tensor:
#         """
#         mean_hat: [B, T,C]
#         returns qhat: [B, 9, T,C] for quantiles [0.1,...,0.9]
#         """
#         mean_hat = mean_hat.transpose(-1, -2)  # [B, C, T]
#         assert mean_hat.dim() == 3, "mean_hat must be [B, C, T]"
#         B, C, T = mean_hat.shape

#         x = mean_hat.reshape(B * C, 1, T)  # share weights across channels
#         x = self.dw(x)
#         x = self.inorm(x)
#         h = self.act(self.pw1(x))
#         o = self.pw2(h)  # [B*C, 9, T]
#         o = o.reshape(B, C, 9, T)

#         bias50 = o[:, :, 0:1, :]  # [B,C,1,T]
#         down_raw = o[:, :, 1:5, :]  # [B,C,4,T]
#         up_raw = o[:, :, 5:9, :]  # [B,C,4,T]

#         # Positive increments -> monotone quantiles
#         down_inc = F.softplus(down_raw) + self.eps
#         up_inc = F.softplus(up_raw) + self.eps

#         q50 = (
#             mean_hat.unsqueeze(2) + bias50
#         )  # allow median to deviate from mean if helpful

#         # cumulative distances from median
#         down_cum = torch.cumsum(down_inc, dim=2)
#         up_cum = torch.cumsum(up_inc, dim=2)

#         q40 = q50 - down_cum[:, :, 0:1, :]
#         q30 = q50 - down_cum[:, :, 1:2, :]
#         q20 = q50 - down_cum[:, :, 2:3, :]
#         q10 = q50 - down_cum[:, :, 3:4, :]

#         q60 = q50 + up_cum[:, :, 0:1, :]
#         q70 = q50 + up_cum[:, :, 1:2, :]
#         q80 = q50 + up_cum[:, :, 2:3, :]
#         q90 = q50 + up_cum[:, :, 3:4, :]

#         qhat = torch.cat(
#             [q10, q20, q30, q40, q50, q60, q70, q80, q90], dim=2
#         )  # [B,C,9,T]

#         qhat = qhat.permute(0, 2, 3, 1)  # [B,9,T,C]

#         return qhat


class TinyTimeMixerEncoder(TinyTimeMixerPreTrainedModel):
    """
    Encoder for TinyTimeMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        if config.multi_scale:
            self.patcher = MultiScaleFromPatchedSequence(config)
        else:
            self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TinyTimeMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config=config)

        if config.resolution_prefix_tuning:
            mid_dim = (config.patch_length + config.d_model) // 2

            self.freq_mod = nn.Sequential(
                nn.Embedding(config.frequency_token_vocab_size, config.patch_length),
                nn.Linear(config.patch_length, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, config.d_model),
            )
        self.resolution_prefix_tuning = config.resolution_prefix_tuning
        self.d_model = config.d_model

        self.add_tokens = None
        self.add_fft_tokens = None
        self.base_norm = None
        if config.register_tokens > 0:
            device = next(self.parameters()).device
            self.add_tokens = TinyTimeMixerAddLearnableRegisterTokens(config, device)

        if config.fft_length > 0:
            self.add_fft_tokens = TinyTimeMixerAddFFTPatches(config)

        if self.config.multi_scale or self.config.enable_base_norm_always:
            self.base_norm = nn.LayerNorm(self.config.num_patches * self.config.d_model, eps=config.norm_eps)

        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    @replace_return_docstrings(output_type=TinyTimeMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        unpatched_past_values: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, TinyTimeMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # flatten [bs x num_patch x d_model]. common_channel/mix_channel: [bs x n_vars x num_patch x d_model]
        patches = self.patcher(past_values)
        if self.resolution_prefix_tuning:
            if freq_token is not None:
                freq_embedding = self.freq_mod(freq_token.long())  # bs x d_model

                freq_embedding = freq_embedding.view(patches.shape[0], 1, 1, self.d_model)
                freq_embedding = freq_embedding.expand(
                    patches.shape[0],
                    patches.shape[1],
                    1,
                    self.d_model,
                )  # bs x channels x 1 x num_features

                patches = torch.cat((freq_embedding, patches), dim=-2)  # bs x channels x num_patch+1 x num_features

            else:
                raise Exception("Expecting freq_token in forward")

        if self.add_tokens is not None:
            patches, _ = self.add_tokens(patches)

        if self.add_fft_tokens is not None:
            patches, _ = self.add_fft_tokens(patches, unpatched_past_values)
        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        if self.base_norm is not None:
            B, C, P, D = patches.shape

            patches = patches.reshape(B, C, P * D)
            patches = self.base_norm(patches)
            patches = patches.reshape(B, C, P, D)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        return TinyTimeMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


@dataclass
class TinyTimeMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The TinyTimeMixer Model for time-series forecasting.",
    TINYTIMEMIXER_START_DOCSTRING,
)
class TinyTimeMixerModel(TinyTimeMixerPreTrainedModel):
    def __init__(self, config: TinyTimeMixerConfig):
        if config.init_processing is False:
            config.check_and_init_preprocessing()

        super().__init__(config)

        self.use_return_dict = config.use_return_dict
        self.encoder = TinyTimeMixerEncoder(config)
        self.patching = TinyTimeMixerPatchify(config)

        if config.scaling == "mean":
            self.scaler = TinyTimeMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TinyTimeMixerStdScaler(config)
        else:
            self.scaler = TinyTimeMixerNOPScaler(config)

        self.d_model = config.d_model

        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    @add_start_docstrings_to_model_forward(TINYTIMEMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TinyTimeMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerModelOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        enc_input = patched_x

        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
            unpatched_past_values=scaled_past_values,
        )

        if isinstance(encoder_output, tuple):
            encoder_output = TinyTimeMixerEncoderOutput(*encoder_output)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    patched_x,
                    loc,
                    scale,
                ]
            )

        return TinyTimeMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            loc=loc,
            scale=scale,
        )


@dataclass
class TinyTimeMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`TinyTimeMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None
    input_data: torch.FloatTensor = None
    forecast_groundtruth: torch.FloatTensor = None
    quantile_outputs: torch.FloatTensor = None


@dataclass
class TinyTimeMixerForDecomposedPredictionOutput(ModelOutput):
    """ """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    quantile_outputs: torch.FloatTensor = None
    trend_prediction_outputs: torch.FloatTensor = None
    residual_prediction_outputs: torch.FloatTensor = None
    trend_input: torch.FloatTensor = None
    residual_input: torch.FloatTensor = None
    trend_quantile_outputs: torch.FloatTensor = None
    residual_quantile_outputs: torch.FloatTensor = None
    input_data: torch.FloatTensor = None
    forecast_groundtruth: torch.FloatTensor = None


@dataclass
class SampleTinyTimeMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


class TinyTimeMixerForPrediction(TinyTimeMixerPreTrainedModel):
    r"""
    `TinyTimeMixer` for forecasting application.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        config.check_and_init_preprocessing()
        super().__init__(config)

        self.config = config

        self.loss = config.loss

        self.use_return_dict = config.use_return_dict

        self.prediction_channel_indices = config.prediction_channel_indices
        self.num_parallel_samples = config.num_parallel_samples

        self.num_input_channels = config.num_input_channels

        self.prediction_filter_length = config.prediction_filter_length

        if config.loss in ["mse", "mae", "pinball", "huber"] or config.loss is None:
            self.distribution_output = None
        elif config.loss == "nll":
            if self.prediction_filter_length is None:
                dim = config.prediction_length
            else:
                dim = config.prediction_filter_length

            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                self.distribution_output = output_class(dim=dim)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.backbone = TinyTimeMixerModel(config)

        self.use_decoder = config.use_decoder

        if config.use_decoder:
            self.decoder = TinyTimeMixerDecoder(config)

        self.head = TinyTimeMixerForPredictionHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        self.multi_quantile_head_block = None
        if config.multi_quantile_head:
            self.multi_quantile_head_block = MultiQuantileHead(config)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(TINYTIMEMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TinyTimeMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForPredictionOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        future_observed_mask (`torch.Tensor` of shape `(batch_size, prediction_length, num_targets)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
        static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`
        metadata (`torch.Tensor`, *optional*): A tensor containing metadata. Currently unused in TinyTimeMixer, but used
            to support custom trainers. Defaults to None.

        Returns:

        """
        if past_values.dim() != 3:
            raise ValueError(
                "`past_values` must have 3 dimensions of shape `(batch_size, sequence_length, num_input_channels)`."
            )

        sequence_length = (
            self.config.masked_context_length
            if self.config.masked_context_length is not None
            else self.config.context_length
        )

        if past_values.shape[1] > sequence_length:
            past_values = past_values[:, -sequence_length:, :]
        elif past_values.shape[1] < sequence_length:
            pad_length = sequence_length - past_values.shape[1]

            # Left-pad zeros in the beginning of the time dimension.
            # Shape: [B, pad_length, C]
            pad_values = past_values.new_zeros(
                past_values.shape[0],
                pad_length,
                *past_values.shape[2:],
            )
            past_values = torch.cat([pad_values, past_values], dim=1)

        # elif past_values.shape[1] < sequence_length:
        #     raise ValueError("Context length in `past_values` is shorter that TTM context_length.")

        if past_observed_mask is not None:
            if past_observed_mask.shape[1] > sequence_length:
                past_observed_mask = past_observed_mask[:, -sequence_length:, :]

            elif past_observed_mask.shape[1] < sequence_length:
                pad_length = sequence_length - past_observed_mask.shape[1]

                # Left-pad mask with zeros/False.
                # This marks padded values as unobserved.
                pad_mask = past_observed_mask.new_zeros(
                    past_observed_mask.shape[0],
                    pad_length,
                    *past_observed_mask.shape[2:],
                )

                past_observed_mask = torch.cat([pad_mask, past_observed_mask], dim=1)

        # if past_observed_mask is not None and past_observed_mask.shape[1] > sequence_length:
        #     past_observed_mask = past_observed_mask[:, -sequence_length:, :]

        if self.multi_quantile_head_block is not None:
            loss = MultiPinballLoss(self.config)
        elif self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "mae":
            loss = nn.L1Loss(reduction="mean")
        elif self.loss == "pinball":
            loss = PinballLoss(quantile=self.config.quantile)
        elif self.loss == "huber":
            loss = nn.HuberLoss(delta=self.config.huber_delta)
        elif self.loss == "nll":
            raise Exception(
                "NLL loss and Distribution heads are currently not allowed. Use mse or mae as loss functions."
            )
            loss = nll
        elif self.loss is None:
            loss = None
        else:
            raise ValueError("Invalid loss function: Allowed values: mse, mae and nll")

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.backbone(
            past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            freq_token=freq_token,
        )  # model_output: [batch_size x nvars x num_patch x d_model]

        if isinstance(model_output, tuple):
            model_output = TinyTimeMixerModelOutput(*model_output)

        decoder_input = model_output.last_hidden_state
        hidden_states = model_output.hidden_states

        if self.use_decoder:
            decoder_output, decoder_hidden_states = self.decoder(
                hidden_state=decoder_input,
                patch_input=model_output.patch_input,
                output_hidden_states=output_hidden_states,
                static_categorical_values=static_categorical_values,
            )  # [batch_size x nvars x num_patch x decoder_d_model]

            if decoder_hidden_states:
                hidden_states.extend(decoder_hidden_states)

        else:
            decoder_output = decoder_input

        # tensor [batch_size x prediction_length x num_input_channels]

        # head should take future mask
        y_hat = self.head(decoder_output, past_values=past_values, future_values=future_values)

        if (
            self.prediction_filter_length is not None
            and future_values is not None
            and future_values.shape[1] != self.prediction_filter_length
        ):
            future_values = future_values[:, : self.prediction_filter_length, :]

            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[:, : self.prediction_filter_length, :]

        if (
            self.prediction_channel_indices is not None
            and future_values is not None
            and future_values.shape[2] != len(self.prediction_channel_indices)
            and future_values.shape[2] == self.num_input_channels
        ):
            future_values = future_values[..., self.prediction_channel_indices]

            if future_observed_mask is not None:
                future_observed_mask = future_observed_mask[..., self.prediction_channel_indices]

        if self.prediction_channel_indices is not None:
            loc = model_output.loc[..., self.prediction_channel_indices]
            scale = model_output.scale[..., self.prediction_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale
        # loc/scale: batch_size x 1 x prediction_channel_indices or num_targets

        y_hat_quantiles = None  # torch.zeros(1, device=y_hat.device)

        loss_val = None
        # y_hat_unscaled = y_hat
        if future_observed_mask is not None:
            fut_mask_bool = future_observed_mask.type(torch.bool)

        if self.distribution_output:
            raise Exception("distribution_output: Deprecated workflow")
            distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
            if future_values is not None and return_loss is True and loss is not None:
                if future_observed_mask is not None and (~fut_mask_bool).any():
                    if (~fut_mask_bool).all():
                        # no valid observed values
                        print(future_observed_mask)
                        raise ValueError("Loss computation failed due to too many missing values")
                    loss_val = loss(distribution, future_values)
                    # select only values of loss where entire timepoint is observed
                    loss_val = loss_val[fut_mask_bool.all(dim=-1)]
                else:
                    loss_val = loss(distribution, future_values)
                loss_val = weighted_average(loss_val)

        elif self.multi_quantile_head_block is not None:
            num_quantiles = self.multi_quantile_head_block.num_quantiles
            median_index = num_quantiles // 2
            y_hat_quantiles = self.multi_quantile_head_block(y_hat, decoder_hidden_state=decoder_output)
            y_hat = y_hat_quantiles[:, median_index, ...]
            # y_hat_unscaled = y_hat
            y_hat = self.backbone.scaler.inverse(data=y_hat, loc=loc, scale=scale)

            loc_expand = loc.unsqueeze(1).repeat(1, num_quantiles, 1, 1)
            scale_expand = scale.unsqueeze(1).repeat(1, num_quantiles, 1, 1)

            y_hat_quantiles = self.backbone.scaler.inverse(data=y_hat_quantiles, loc=loc_expand, scale=scale_expand)

            if future_values is not None and return_loss is True and loss is not None:
                yq = y_hat_quantiles
                yt = future_values
                yp = y_hat

                loss_val = loss(yq, yt, horizon_weights=None) + float(
                    getattr(self.config, "point_extra_weight", 0.0)
                ) * weighted_l1_over_horizon(yp, yt, horizon_weights=None)

        else:
            y_hat = self.backbone.scaler.inverse(data=y_hat, loc=loc, scale=scale)
            if future_values is not None and return_loss is True and loss is not None:
                if future_observed_mask is not None:
                    loss_val = loss(y_hat[fut_mask_bool], future_values[fut_mask_bool])
                else:
                    # avoiding mask operations for performance benefits on normal scenarios.
                    loss_val = loss(y_hat, future_values)

        m_last_hidden_state = model_output.last_hidden_state

        if self.config.light_mode:
            m_last_hidden_state = None
            decoder_output = None
            hidden_states = None
            loc = None
            scale = None
            past_values = None
            future_values = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    m_last_hidden_state,
                    decoder_output,
                    hidden_states,
                    loc,
                    scale,
                    past_values,
                    future_values,
                    y_hat_quantiles,
                ]
            )

        return TinyTimeMixerForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x prediction_length x num_input_channels]
            backbone_hidden_state=m_last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            decoder_hidden_state=decoder_output,  # x: [batch_size x nvars x num_patch x decoder_d_model]
            hidden_states=hidden_states,
            loc=loc,
            scale=scale,
            input_data=past_values,
            forecast_groundtruth=future_values,
            quantile_outputs=y_hat_quantiles,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SampleTinyTimeMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]` or `[False, True]`:
                    - 1 or True for values that are **observed**,
                    - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SampleTinyTimeMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        # get number of samples
        num_parallel_samples = self.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution

        distribution = self.distribution_output.distribution(
            outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
        )

        # get samples: list of [batch_size x prediction_length x num_channels]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]

        # stack tensors
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x prediction_length x num_channels]
        return SampleTinyTimeMixerPredictionOutput(sequences=samples)


class TinyTimeMixerForMaskedPrediction(TinyTimeMixerForPrediction):
    def __init__(self, config: TinyTimeMixerConfig):
        if config.prediction_filter_length is not None:
            append_length = config.prediction_filter_length
        else:
            append_length = config.prediction_length

        self.append_length = append_length
        config.masked_context_length = config.context_length + append_length
        config.fcm_prepend_past_offset = append_length

        if config.exogenous_channel_indices is not None:
            self.non_exog_channels = list(
                set(range(config.num_input_channels)) - set(config.exogenous_channel_indices)
            )
        else:
            self.non_exog_channels = list(range(config.num_input_channels))

        super().__init__(config)

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForPredictionOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        future_observed_mask (`torch.Tensor` of shape `(batch_size, prediction_length, num_targets)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
        static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`
        metadata (`torch.Tensor`, *optional*): A tensor containing metadata. Currently unused in TinyTimeMixer, but used
            to support custom trainers. Defaults to None.

        Returns:

        """
        if future_values is not None:
            future_values_masked = future_values.clone()
        else:
            future_values_masked = torch.zeros(past_values.shape[0], self.append_length, past_values.shape[2])

        if (
            self.config.prediction_filter_length is not None
            and future_values_masked is not None
            and future_values_masked.shape[1] != self.config.prediction_filter_length
        ):
            future_values_masked = future_values_masked[:, : self.config.prediction_filter_length, :]

        if self.config.exogenous_channel_indices is not None:
            future_values_masked[:, :, self.non_exog_channels] = self.config.mask_value
        else:
            future_values_masked.fill_(self.config.mask_value)
        past_values = torch.cat((past_values, future_values_masked), dim=-2)  # xb: [bs x seq_len+ fl x n_vars]

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # if there is already a past mask - update with it
        # index 1 refers to the seq len

        if past_observed_mask.shape[1] < past_values.shape[1]:
            temp_mask = torch.ones_like(past_values)
            temp_mask[:, : past_observed_mask.shape[1], :] = past_observed_mask
            past_observed_mask = temp_mask

        # past_observed_mask[:, -self.config.prediction_length :, :] = 0
        past_observed_mask[:, -self.config.prediction_length :, self.non_exog_channels] = 0
        # [bs x seq_len+ fl x n_vars]

        return super().forward(
            past_values=past_values,
            future_values=future_values,
            past_observed_mask=past_observed_mask,
            future_observed_mask=future_observed_mask,
            output_hidden_states=output_hidden_states,
            return_loss=return_loss,
            return_dict=return_dict,
            freq_token=freq_token,
            static_categorical_values=static_categorical_values,
            metadata=metadata,
        )


# ---------------------------
# Gaussian kernel helper
# ---------------------------
def _gaussian_kernel(w: int, device=None, dtype=None, sigma: Optional[float] = None) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel of length w (odd), normalized to sum=1.
    sigma default is proportional to window, giving a smooth low-pass.
    """
    assert w % 2 == 1 and w >= 3, "w must be odd and >=3"
    if sigma is None:
        # A gentle default; ~95% mass inside window
        sigma = 0.25 * w
    r = (w - 1) // 2
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / (k.sum() + 1e-12)).view(1, 1, w)  # [1,1,w]
    return k


def robust_lowess_like(
    x: torch.Tensor,
    frac: float = 0.15,
    iters: int = 2,
    eps: float = 1e-8,
    pad_mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust LOWESS-like smoothing via Gaussian local average + Tukey bisquare reweighting.

    Args:
        x        : [B, T, C] float tensor
        frac     : fraction of T for local window length (w = round(frac*T), forced odd)
        iters    : robust reweighting iterations
        eps      : numerical stability
        pad_mode : F.pad mode for time padding

    Returns:
        trend    : [B, T, C]
        residual : [B, T, C]
    """
    assert x.dim() == 3, "x must be [B, T, C]"
    B, T, C = x.shape
    device, dtype = x.device, x.dtype

    # choose odd window length (bounded)
    w = max(5, int(round(frac * T)))
    if w % 2 == 0:
        w += 1
    w = min(w, T if T % 2 == 1 else T - 1)  # ensure w <= T and odd

    # gaussian kernel
    k = _gaussian_kernel(w, device=device, dtype=dtype)  # [1,1,w]

    def _gauss_conv(y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # y: [B,T,C], weights: [B,T,C] or None
        y_bc_t = y.permute(0, 2, 1).contiguous().reshape(B * C, 1, T)  # [BC,1,T]
        if weights is None:
            num = F.conv1d(F.pad(y_bc_t, (w // 2, w // 2), mode=pad_mode), k)
            den = F.conv1d(F.pad(torch.ones_like(y_bc_t), (w // 2, w // 2), mode=pad_mode), k)
        else:
            wts_bc_t = weights.permute(0, 2, 1).contiguous().reshape(B * C, 1, T)
            num = F.conv1d(F.pad(y_bc_t * wts_bc_t, (w // 2, w // 2), mode=pad_mode), k)
            den = F.conv1d(F.pad(wts_bc_t, (w // 2, w // 2), mode=pad_mode), k)
        out = num / (den + eps)
        return out.view(B, C, T).permute(0, 2, 1).contiguous()  # [B,T,C]

    # initial smooth (no robust weights)
    trend = _gauss_conv(x, weights=None)
    # DC guard: keep per-series mean aligned
    trend = trend - trend.mean(dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

    # robust reweighting loops (Tukey bisquare on centered residuals with MAD scale)
    for _ in range(iters):
        resid = x - trend
        resid_med = resid.median(dim=1, keepdim=True).values  # [B,1,C]
        resid_c = resid - resid_med
        mad = torch.median(torch.abs(resid_c), dim=1, keepdim=True).values + eps  # [B,1,C]
        u = resid_c / (6.0 * mad)  # scaled residuals
        wts = (1.0 - torch.clamp(u**2, 0.0, 1.0)) ** 2  # Tukey's bisquare ∈ [0,1]

        trend = _gauss_conv(x, weights=wts)
        trend = trend - trend.mean(dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

    residual = x - trend
    return trend, residual


class TinyTimeMixerForDecomposedPrediction(TinyTimeMixerPreTrainedModel):
    r"""
    `TinyTimeMixer` for forecasting application.

    Args:
        config (`TinyTimeMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TinyTimeMixerConfig):
        self._init_decomposed(config)

    def _init_decomposed(self, config: TinyTimeMixerConfig):
        super().__init__(config)

        trend_config = copy.deepcopy(config)

        if config.trend_patch_length is not None:
            trend_config.patch_length = config.trend_patch_length
        if config.trend_patch_stride is not None:
            trend_config.patch_stride = config.trend_patch_stride
        if config.trend_d_model is not None:
            trend_config.d_model = config.trend_d_model
        if config.trend_decoder_d_model is not None:
            trend_config.decoder_d_model = config.trend_decoder_d_model
        if config.trend_num_layers is not None:
            trend_config.num_layers = config.trend_num_layers
        if config.trend_decoder_num_layers is not None:
            trend_config.decoder_num_layers = config.trend_decoder_num_layers
        if config.trend_register_tokens is not None:
            trend_config.register_tokens = config.trend_register_tokens
        if config.trend_fft_length is not None:
            trend_config.fft_length = config.trend_fft_length

        if config.trend_multi_scale is not None:
            trend_config.multi_scale = config.trend_multi_scale
        if config.trend_adaptive_patching_levels is not None:
            trend_config.adaptive_patching_levels = config.trend_adaptive_patching_levels
        if config.trend_head_d_model is not None:
            trend_config.head_d_model = config.trend_head_d_model

        trend_config.num_patches = None
        trend_config.scaling = None  # "std"

        residual_config = copy.deepcopy(config)

        if config.residual_context_length is not None:
            residual_config.context_length = config.residual_context_length

        residual_config.scaling = None  # "std"

        self.use_return_dict = config.use_return_dict

        if config.scaling == "mean":
            self.scaler = TinyTimeMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TinyTimeMixerStdScaler(config)
        else:
            self.scaler = TinyTimeMixerNOPScaler(config)

        self.trend_forecaster = TinyTimeMixerForPrediction(trend_config)

        self.residual_forecaster = TinyTimeMixerForPrediction(residual_config)

        self.prediction_channel_indices = config.prediction_channel_indices

        self.prediction_filter_length = config.prediction_filter_length

        self.loss = config.loss

        self.config = config

        self.multi_quantile_head = config.multi_quantile_head
        self.num_input_channels = config.num_input_channels
        self.forecast_loss_type = config.forecast_loss_type
        self.trend_loss_weight = config.trend_loss_weight
        self.residual_loss_weight = config.residual_loss_weight
        self.joint_loss_weight = config.joint_loss_weight

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForDecomposedPredictionOutput:
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        return self._forward_decomposed(
            past_values=past_values,
            future_values=future_values,
            past_observed_mask=past_observed_mask,
            future_observed_mask=future_observed_mask,
            output_hidden_states=output_hidden_states,
            return_loss=return_loss,
            return_dict=return_dict,
            freq_token=freq_token,
            static_categorical_values=static_categorical_values,
            metadata=metadata,
        )

    def set_stage(self, forecast_loss_type: str, w_tr: float, w_res: float, w_joint: float):
        print(f"[MODEL] set_stage → {forecast_loss_type}  weights=({w_tr},{w_res},{w_joint})")
        self.forecast_loss_type = forecast_loss_type
        self.trend_loss_weight = float(w_tr)
        self.residual_loss_weight = float(w_res)
        self.joint_loss_weight = float(w_joint)

    def _choose_loss(self):
        if self.multi_quantile_head:
            return MultiPinballLoss(self.config)
        if self.loss == "mse":
            return nn.MSELoss(reduction="mean")
        elif self.loss == "mae":
            return nn.L1Loss(reduction="mean")
        elif self.loss == "pinball":
            return PinballLoss(quantile=self.config.quantile)
        elif self.loss == "huber":
            return nn.HuberLoss(delta=self.config.huber_delta)
        elif self.loss is None:
            return None
        else:
            raise ValueError("Invalid loss function: Allowed values: mse, mae, huber, pinball")

    def _build_targets(
        self,
        past_scaled: torch.Tensor,
        future_scaled: torch.Tensor,
    ):
        """
        Build trend/residual targets from teacher on [past ⊕ future] (zero-phase).
        Returns:
          tau_tgt [B, L_fut, C], r_tgt [B, L_fut, C]
        """

        with torch.no_grad():
            # concat along time: [B, L_ctx+L_fut, C]
            x_full = torch.cat([past_scaled, future_scaled], dim=1)
            tau_full, _ = robust_lowess_like(x_full)  # returns (trend,resid) on full
            L_fut = future_scaled.size(1)
            tau_tgt = tau_full[:, -L_fut:, :]  # [B,L_fut,C]
            r_tgt = future_scaled - tau_tgt  # [B,L_fut,C]
        return tau_tgt, r_tgt

    def _detrend_short_ctx(self, past_scaled: torch.Tensor, tau_ctx_est: torch.Tensor):
        """
        Use model context trend (stop-grad) to detrend the *short* residual context.
        past_scaled : [B, L_ctx, C]
        tau_ctx_est : [B, L_ctx, C] (model estimate over context)
        Returns:
          x_res_ctx  : [B, L_res_ctx, C]
        """
        B, L_ctx, C = past_scaled.shape
        L_res = (
            min(self.config.residual_context_length, L_ctx)
            if self.config.residual_context_length is not None
            else L_ctx
        )
        # L_res = min(self.config.residual_context_length, L_ctx)
        x_short = past_scaled[:, -L_res:, :]  # [B,L_res,C]
        tau_tail = tau_ctx_est[:, -L_res:, :].detach()  # stop-grad
        return x_short - tau_tail, L_res  # [B,L_res,C]

    def combine_quantiles_fast(self, trend_q, resid_q, eps: float = 1e-8):
        if trend_q is None and resid_q is None:
            return None
        if trend_q is None:
            return resid_q
        if resid_q is None:
            return trend_q

        B, Q, L, C = trend_q.shape
        k = Q // 2

        trend_q50 = trend_q[:, k : k + 1, :, :]
        resid_q50 = resid_q[:, k : k + 1, :, :]
        combined_q50 = trend_q50 + resid_q50

        trend_width = trend_q - trend_q50
        resid_width = resid_q - resid_q50

        # Stable sqrt (prevents sqrt(0) backward blow-up)
        x = trend_width.square() + resid_width.square()
        combined_width_mag = torch.sqrt(x + eps)

        sign = torch.sign(trend_width + resid_width)
        combined_width = sign * combined_width_mag

        # Force exact median width to 0 WITHOUT in-place ops
        # mask shape: [1,Q,1,1] broadcast to [B,Q,L,C]
        q_idx = torch.arange(Q, device=trend_q.device).view(1, Q, 1, 1)
        median_mask = q_idx == k
        combined_width = torch.where(median_mask, torch.zeros_like(combined_width), combined_width)

        combined_q = combined_q50 + combined_width
        return combined_q

    # def combine_quantiles_fast(
    #     self,
    #     trend_q: torch.Tensor,  # [B,Q,L,C]
    #     resid_q: torch.Tensor,  # [B,Q,L,C]
    # ):
    #     """
    #     Fast deterministic quantile combination using sqrt-of-squares width rule.

    #     Assumptions:
    #     - Q is odd
    #     - Median is at center index (Q // 2)
    #     - Same quantile ordering for both models

    #     Returns:
    #     combined_q: [B,Q,L,C]
    #     """

    #     if trend_q is None and resid_q is None:
    #         return None

    #     B, Q, L, C = trend_q.shape
    #     k = Q // 2  # median index

    #     # ----------------------------
    #     # 1) Combine median
    #     # ----------------------------
    #     trend_q50 = trend_q[:, k : k + 1, :, :]  # [B,1,L,C]
    #     resid_q50 = resid_q[:, k : k + 1, :, :]  # [B,1,L,C]

    #     combined_q50 = trend_q50 + resid_q50  # [B,1,L,C]

    #     # ----------------------------
    #     # 2) Compute widths relative to median
    #     # ----------------------------
    #     trend_width = trend_q - trend_q50  # [B,Q,L,C]
    #     resid_width = resid_q - resid_q50  # [B,Q,L,C]

    #     # sqrt-of-squares combination
    #     combined_width_mag = torch.sqrt(trend_width.pow(2) + resid_width.pow(2))

    #     # Preserve side (lower vs upper)
    #     sign = torch.sign(trend_width + resid_width)

    #     combined_width = sign * combined_width_mag  # [B,Q,L,C]

    #     # ----------------------------
    #     # 3) Add back median
    #     # ----------------------------
    #     combined_q = combined_q50 + combined_width

    #     # Ensure exact median (avoid tiny numerical drift)
    #     combined_q[:, k : k + 1, :, :] = combined_q50

    #     return combined_q

    def _forward_decomposed(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> TinyTimeMixerForDecomposedPredictionOutput:
        r"""
        past_observed_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        future_observed_mask (`torch.Tensor` of shape `(batch_size, prediction_length, num_targets)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]` or `[False, True]`:
                - 1 or True for values that are **observed**,
                - 0 or False for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
        static_categorical_values (`torch.FloatTensor` of shape `(batch_size, number_of_categorical_variables)`, *optional*):
            Tokenized categorical values can be passed here. Ensure to pass in the same order as the vocab size list used in the
            TinyTimeMixerConfig param `categorical_vocab_size_list`
        metadata (`torch.Tensor`, *optional*): A tensor containing metadata. Currently unused in TinyTimeMixer, but used
            to support custom trainers. Defaults to None.

        Returns:

        """

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        with torch.no_grad():
            trend_signal, _ = robust_lowess_like(scaled_past_values)

        residual_signal, L_res = self._detrend_short_ctx(scaled_past_values, trend_signal)

        residual_observed_mask = past_observed_mask[:, -L_res:, :].contiguous()

        trend_prediction = self.trend_forecaster(
            past_values=scaled_past_values,
            # past_values=trend_signal,
            future_values=future_values,
            past_observed_mask=past_observed_mask,
            future_observed_mask=future_observed_mask,
            output_hidden_states=output_hidden_states,
            return_loss=False,
            return_dict=return_dict,
            freq_token=freq_token,
            static_categorical_values=static_categorical_values,
            metadata=metadata,
        )

        residual_prediction = self.residual_forecaster(
            past_values=residual_signal,
            future_values=future_values,
            past_observed_mask=residual_observed_mask,
            future_observed_mask=future_observed_mask,
            output_hidden_states=output_hidden_states,
            return_loss=False,
            return_dict=return_dict,
            freq_token=freq_token,
            static_categorical_values=static_categorical_values,
            metadata=metadata,
        )

        trend_prediction_outputs = trend_prediction.prediction_outputs
        residual_prediction_outputs = residual_prediction.prediction_outputs
        combined_point_forecast = trend_prediction_outputs + residual_prediction_outputs

        trend_quantile_outputs = trend_prediction.quantile_outputs
        residual_quantile_outputs = residual_prediction.quantile_outputs

        if trend_quantile_outputs is None or residual_quantile_outputs is None:
            combined_quantile_forecast = None
        else:
            if self.config.combine_quantiles_via_variance:
                combined_quantile_forecast = self.combine_quantiles_fast(
                    trend_quantile_outputs, residual_quantile_outputs
                )
            else:
                combined_quantile_forecast = trend_quantile_outputs + residual_quantile_outputs

        loss_val = None

        tau_tgt = r_tgt = None
        base_loss = self._choose_loss()

        if self.prediction_channel_indices is not None:
            loc = loc[..., self.prediction_channel_indices]
            scale = scale[..., self.prediction_channel_indices]
            trend_signal = trend_signal[..., self.prediction_channel_indices]
            residual_signal = residual_signal[..., self.prediction_channel_indices]
            scaled_past_values = scaled_past_values[..., self.prediction_channel_indices]

        if future_values is not None and return_loss and base_loss is not None:
            if self.prediction_filter_length is not None and future_values.shape[1] != self.prediction_filter_length:
                future_values = future_values[:, : self.prediction_filter_length, :]

                if future_observed_mask is not None:
                    future_observed_mask = future_observed_mask[:, : self.prediction_filter_length, :]

            if (
                self.prediction_channel_indices is not None
                and future_values.shape[2] != len(self.prediction_channel_indices)
                and future_values.shape[2] == self.num_input_channels
            ):
                future_values = future_values[..., self.prediction_channel_indices]

                if future_observed_mask is not None:
                    future_observed_mask = future_observed_mask[..., self.prediction_channel_indices]

            # scale future with SAME (loc, scale) used for past
            scaled_future_values = self.scaler.transform(future_values, loc, scale)
            tau_tgt, r_tgt = self._build_targets(scaled_past_values, scaled_future_values)  # [B,L_fut,C] each

            if self.multi_quantile_head:
                # pinball on quantiles vs raw future
                point_extra_weight = self.config.point_extra_weight
                joint_loss = base_loss(
                    combined_quantile_forecast, scaled_future_values
                ) + point_extra_weight * F.l1_loss(combined_point_forecast, scaled_future_values)
                trend_loss = (
                    base_loss(trend_quantile_outputs, tau_tgt)
                    + 0.1
                    * F.l1_loss(
                        trend_prediction_outputs[:, 1:, :] - trend_prediction_outputs[:, :-1, :],
                        tau_tgt[:, 1:, :] - tau_tgt[:, :-1, :],
                    )
                    + point_extra_weight * F.l1_loss(trend_prediction_outputs, tau_tgt)
                )
                residual_loss = base_loss(residual_quantile_outputs, r_tgt) + point_extra_weight * F.l1_loss(
                    residual_prediction_outputs, r_tgt
                )

            else:
                joint_loss = base_loss(combined_point_forecast, scaled_future_values)

                trend_loss = base_loss(trend_prediction_outputs, tau_tgt) + 0.1 * base_loss(
                    trend_prediction_outputs[:, 1:, :] - trend_prediction_outputs[:, :-1, :],
                    tau_tgt[:, 1:, :] - tau_tgt[:, :-1, :],
                )

                residual_loss = base_loss(residual_prediction_outputs, r_tgt)

            loss_val = (
                self.joint_loss_weight * joint_loss
                + self.trend_loss_weight * trend_loss
                + self.residual_loss_weight * residual_loss
            )

        # inverse to get back original scale
        combined_point_forecast = self.scaler.inverse(data=combined_point_forecast, loc=loc, scale=scale)

        trend_prediction_outputs = self.scaler.inverse(data=trend_prediction_outputs, loc=loc, scale=scale)

        residual_prediction_outputs = self.scaler.inverse(data=residual_prediction_outputs, loc=loc, scale=scale)

        trend_signal = self.scaler.inverse(data=trend_signal, loc=loc, scale=scale)

        residual_signal = self.scaler.inverse(data=residual_signal, loc=loc, scale=scale)

        if self.multi_quantile_head:
            num_quantiles = self.trend_forecaster.multi_quantile_head_block.num_quantiles
            loc_exp = loc.unsqueeze(1).repeat(1, num_quantiles, 1, 1)
            scale_exp = scale.unsqueeze(1).repeat(1, num_quantiles, 1, 1)

            # y_hat_quantiles = y_hat_quantiles.reshape(-1, s, c)
            combined_quantile_forecast = self.scaler.inverse(
                data=combined_quantile_forecast, loc=loc_exp, scale=scale_exp
            )
            trend_quantile_outputs = self.scaler.inverse(data=trend_quantile_outputs, loc=loc_exp, scale=scale_exp)
            residual_quantile_outputs = self.scaler.inverse(
                data=residual_quantile_outputs, loc=loc_exp, scale=scale_exp
            )

        if self.config.light_mode:
            trend_prediction_outputs = None
            residual_prediction_outputs = None
            trend_signal = None
            residual_signal = None
            trend_quantile_outputs = None
            residual_quantile_outputs = None
            past_values = None
            future_values = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    combined_point_forecast,  # tensor [batch_size x prediction_length x num_input_channels]
                    combined_quantile_forecast,
                    trend_prediction_outputs,
                    residual_prediction_outputs,
                    trend_signal,
                    residual_signal,
                    trend_quantile_outputs,
                    residual_quantile_outputs,
                    past_values,
                    future_values,
                ]
            )

        return TinyTimeMixerForDecomposedPredictionOutput(
            loss=loss_val,
            prediction_outputs=combined_point_forecast,  # tensor [batch_size x prediction_length x num_input_channels]
            quantile_outputs=combined_quantile_forecast,
            trend_prediction_outputs=trend_prediction_outputs,
            residual_prediction_outputs=residual_prediction_outputs,
            trend_input=trend_signal,
            residual_input=residual_signal,
            trend_quantile_outputs=trend_quantile_outputs,
            residual_quantile_outputs=residual_quantile_outputs,
            input_data=past_values,
            forecast_groundtruth=future_values,
        )
