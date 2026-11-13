# ruff: noqa
"""Trimmed from utils/input_preprocessing.py: the inference normalization only.

Verbatim apart from dropping the PatchTSTFMConfig type annotation; `cfg` needs
only `.eps`, `.patch_size` and `.num_patches`.
"""

from typing import Tuple

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


def mask_aware_normalize_for_inference(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize using only visible historical values.

    Robust GIFT-Eval handling
    -------------------------
    Some GIFT-Eval rolling windows, especially after multivariate-to-univariate
    expansion, can contain channels with zero or one finite historical value.
    We should still return a valid forecast instead of aborting evaluation.

    Let

        n_i = number of visible finite historical values in series i.

    We use:
      - n_i >= 2: usual mask-aware mean/std;
      - n_i = 1 : mean = the single observed value, std = 1;
      - n_i = 0 : mean = 0, std = 1.

    This keeps forecasts finite and lets GluonTS mask invalid labels during
    metric computation.
    """
    missing_mask = (~observed_mask) & (~padding_mask) & (~pred_mask)
    union_mask = pred_mask | missing_mask | padding_mask

    visible_mask = observed_mask & (~pred_mask) & (~padding_mask)
    count = visible_mask.sum(dim=1, keepdim=True).to(x.dtype)

    x_visible = torch.where(visible_mask, x, torch.zeros_like(x))
    safe_count = count.clamp_min(1.0)

    raw_mean = x_visible.sum(dim=1, keepdim=True) / safe_count
    mean = torch.where(count > 0, raw_mean, torch.zeros_like(raw_mean))

    raw_var = (
        torch.where(visible_mask, (x - mean).pow(2), torch.zeros_like(x)).sum(
            dim=1, keepdim=True
        )
        / safe_count
    )

    # If there are fewer than two visible points, variance is not identifiable.
    # Use unit scale rather than sqrt(eps), because sqrt(eps) would make the
    # reverse normalization almost constant and numerically brittle.
    std = torch.where(
        count >= 2, torch.sqrt(raw_var + cfg.eps), torch.ones_like(raw_var)
    )

    x_norm_all = torch.asinh((x - mean) / std)
    x_norm_input = torch.where(union_mask, torch.zeros_like(x_norm_all), x_norm_all)

    B, T = x.shape
    L, N = cfg.patch_size, cfg.num_patches
    patch_padding = padding_mask.reshape(B, N, L).all(dim=-1)

    return x_norm_input, mean, std, union_mask, patch_padding
